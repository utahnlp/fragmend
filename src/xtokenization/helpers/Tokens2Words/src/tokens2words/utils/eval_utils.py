import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import default_data_collator
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import logging

from .bits_per_byte import BitsPerByte
logger = logging.getLogger(__name__)

def is_not_number(s):
    try:
        float(s)  # Try converting the string to a float
        return False  # If conversion is successful, it's a number
    except ValueError:
        return True  # If conversion fails, it's not a number


def get_contexts_ending_with_word(word, dataset):
    result_contexts = []
    word_len = len(word)

    # Iterate over the dataset
    for example in dataset:
        text = example["text"]

        # Find all occurrences of the word in the text
        start = 0
        while True:
            idx = text.find(word, start)
            if idx == -1:
                break

            # Ensure that the word is isolated (not a substring of another word)
            if (idx == 0 or not text[idx - 1].isalnum()) and (
                    idx + word_len == len(text) or not text[idx + word_len].isalnum()):
                # Text ends with the word
                result_contexts.append(text[:idx + word_len].strip())
            start = idx + word_len

    return result_contexts


def get_texts_containing_word(words, dataset):
    result_texts = []
    words_set = set(words)

    # Iterate over the dataset
    for example in dataset:
        if words_set.intersection(set(example["text"].split())):
            result_texts.append(example["text"])

    return result_texts


def compute_topk_token_rank(logits, labels, k=1000):
    # Get the top-k predicted logits and their indices
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)

    # Expand the labels for comparison
    labels_expanded = labels.unsqueeze(-1).expand_as(topk_indices)

    # Check if the label token is within the top-k predictions
    rank_in_topk = (topk_indices == labels_expanded).nonzero(as_tuple=False)

    # Create a rank tensor initialized with k (max rank is k)
    ranks = torch.full(labels.shape, k, dtype=torch.long, device=logits.device)

    # For labels in top-k, set the rank accordingly
    ranks[rank_in_topk[:, 0], rank_in_topk[:, 1]] = rank_in_topk[:, 2] + 1

    return ranks


def count_tokens_in_dataset(dataset, tokenizer, text_column='text'):
    def tokenize_and_count(examples):
        return {'num_tokens': [len(tokenizer(ex).input_ids) for ex in examples[text_column]]}

    tokenized_dataset = dataset.map(tokenize_and_count, batched=True, remove_columns=dataset.column_names)
    
    total_tokens = sum(tokenized_dataset['num_tokens'])
    return total_tokens, tokenized_dataset['num_tokens']


def filter_single_token_words(array, tokenizer, add_space_prefix_for_lower=True):
    def _is_multi_token(word):
        if add_space_prefix_for_lower and word[0].islower():
            word = " " + word
        return len(tokenizer.encode(word, add_special_tokens=False))
    token_counts = array.apply(_is_multi_token)
    mask = token_counts > 1
    return array[mask], token_counts


# TODO make clearer what's its use
def get_last_zero_in_every_seq_mask(tensor):
    # Find where consecutive zeros end
    zero_mask = (tensor == 0)
    diff = torch.diff(zero_mask.int(), dim=1)
    last_zero_mask = torch.cat([diff, torch.ones(tensor.size(0), 1, dtype=diff.dtype).to(tensor.device)], dim=1) == -1

    # Create the output
    output = 1 - tensor
    output[zero_mask & ~last_zero_mask] = 0
    return output


def get_first_zero_in_every_seq_mask(tensor):
    # Identify where consecutive zeros begin
    zero_mask = (tensor == 0)
    # Essentially marks the beginning of each sequence of zeros and if 
    # the first element is zero it also marks that (Zeros will be one)
    diff = torch.diff(zero_mask.int(), dim=1, prepend=torch.zeros(tensor.size(0), 1, dtype=torch.int).to(tensor.device))
    first_zero_mask = diff == 1  # Marks the beginning of each sequence of zeros

    # Create the output
    output = 1 - tensor # MArks all zeros
    output[zero_mask & ~first_zero_mask] = 0    # Sets all but the first zero in each sequence to 0
    return output


def _add_start_token(batch, tokenizer):
    bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
    batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
    batch["attention_mask"] = torch.cat(
        [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
    return batch


def _ignore_new_words_in_attention_mask(shift_attention_mask_batch, shift_labels, new_token_ids=None, replaced_token_seqs_by_len=None):
    # Ignore token_ids of new vocabulary words in shift_labels and shift_logits
    if new_token_ids is not None:
        ignore_mask = torch.isin(shift_labels, new_token_ids)
        shift_attention_mask_batch = shift_attention_mask_batch * (~ignore_mask).long()

    # Ignore multi-token sequences of that were replaced with a single token
    if replaced_token_seqs_by_len is not None:
        # Create a mask that will be updated where sequences match
        ignore_mask = shift_attention_mask_batch.clone()  # Clone the attention mask to modify it
        # Loop over sequences in skip_token_seqs
        for seq_len, seqs in replaced_token_seqs_by_len.items():
            # Create a sliding window of the same size as the skip_seq and check for matches
            for i in range(shift_labels.size(1) - seq_len + 1):
                # Check if the sequence matches at position i
                window = shift_labels[:, i:i + seq_len] #batch_size x length
                # seqs dimension: seqs x length; where seqs is the number of sequences of length 'length'
                curr_mask = torch.all(window.unsqueeze(1) == seqs.unsqueeze(0), dim=-1)
                # curr_mask will be a batch_size x seqs, a boolean tensor
                # which matches every subsequence in input to each of the sequences
                # in seqs
                # Check if there was a single match
                if curr_mask.any():
                    # Zero out the ignore mask for the length of the sequence
                    # for the batch item that has a match
                    ignore_mask[curr_mask.any(dim=-1), i:i + seq_len] = 0
        # Apply the ignore mask to the attention mask
        shift_attention_mask_batch *= ignore_mask

    return shift_attention_mask_batch, ignore_mask


# TODO consider not aggregating results here, to enable metrics for specific words
def compute_metrics(
        logits, labels, attention_mask,
        compute_target_metrics=True, compute_subsequent_metrics=True, compute_perplexity=False,
        return_successful_targets=False,
        original_labels=None, original_logits=None,
        debug=False,
        testing_mode=False):
    target_results = dict()  # will hold metrics for all the new words we add or their original tokenization
    background_results = dict()  # will hold metrics for all background tokens, i.e., not the ones we add or replace
    overall_results = dict()  # will hold metrics for all tokens
    successful_targets = None  # will hold list of target tokens successfully predicted
    if compute_subsequent_metrics:
        # prepare labels and attentions masks for computing metrics only for the 1st tokens following the new words
        subsequent_labels = labels[:,  1:]
        subsequent_attention_mask = get_last_zero_in_every_seq_mask(attention_mask[..., :-1].contiguous())
        subsequent_attention_mask_bool = subsequent_attention_mask == 1
    attention_mask_bool = attention_mask == 1   # Convert 0/1 mask to boolean mask
    overall_mask_bool = attention_mask_bool

    if compute_target_metrics:
        # Target mask marks only the positions of the new words added or 
        # the first tokens of the original tokenization
        # Baseline: First token of the original encoding of a newly added token is set to 1
        # Expanded: All newly added tokens are set to 1
        target_mask = get_first_zero_in_every_seq_mask(attention_mask)
        target_mask_bool = target_mask == 1
        # Target mask plus the non-affected tokens have attention 1
        overall_mask_bool = attention_mask_bool | target_mask_bool

    if compute_perplexity:
        background_results["perplexity"] = torch.exp(
            (F.cross_entropy(logits.transpose(1, 2), labels, reduction="none") * attention_mask).sum(1)
            / attention_mask.sum(1)
        ).mean().detach().cpu().numpy()

    top1 = logits.argmax(dim=-1)
    if original_logits is not None:
        orig_top1 = original_logits.argmax(dim=-1)

    if compute_target_metrics:
        # Here there are two parts first we match the labels and predictions
        # Then we only select the entries corresponding to 1s in the mask
        # The output is hence a list of label matches for the relevant tokens
        target_results["top1_acc"] = ((labels == top1)[target_mask_bool]).detach().cpu().numpy()
        if original_labels is not None:
            # Computes whether prediction matches either the new token or the first token of the original encoding
            # Naturally, this cannot be computed for the baseline system as it’ll be the same as “top_1_acc”
            target_results["sum_top1_acc"] = (
                ((original_labels == top1) | (labels == top1))[target_mask_bool]).detach().cpu().numpy()
            if original_logits is not None:
                # Computes correctness of prediction when the new tokens were ignored, is the highest probability then allotted to first token of the corresponding original encoding. 
                # Again, this will be the same as “top_1_acc” for the baseline systems
                target_results["orig_top1_acc"] = (
                    (original_labels == orig_top1)[target_mask_bool]).detach().cpu().numpy()

        if return_successful_targets:
            successful_targets = (labels[(labels == top1) & target_mask_bool]).detach().cpu().numpy()
    # For all the unaffected tokens
    background_results["top1_acc"] = ((
                         labels == top1)[attention_mask_bool]).detach().cpu().numpy()
    if compute_subsequent_metrics:
        background_results["subsequent_top1_acc"] = ((subsequent_labels == top1[:, 1:])[subsequent_attention_mask_bool]).detach().cpu().numpy()
    if original_logits is not None and original_labels is not None:
        background_results["orig_top1_acc"] = (
            (original_labels == orig_top1)[attention_mask_bool]).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["orig_subsequent_top1_acc"] = (
            (subsequent_labels == orig_top1[:, 1:])[subsequent_attention_mask_bool]).detach().cpu().numpy()

    overall_results["top1_acc"] = ((labels == top1))[overall_mask_bool].detach().cpu().numpy()
    if original_labels is not None:
        # Computes whether prediction matches either the new token or the first token of the original encoding
        # Naturally, this cannot be computed for the baseline system as it’ll be the same as “top_1_acc”
        overall_results["sum_top1_acc"] = (
            ((original_labels == top1) | (labels == top1)))[overall_mask_bool].detach().cpu().numpy()
        if original_logits is not None:
            # Computes correctness of prediction when the new tokens were ignored, is the highest probability then allotted to first token of the corresponding original encoding. 
            # Again, this will be the same as “top_1_acc” for the baseline systems
            overall_results["orig_top1_acc"] = (
                (original_labels == orig_top1)[overall_mask_bool]).detach().cpu().numpy()

    if debug:
        import pdb; pdb.set_trace()
    return background_results, target_results, overall_results, successful_targets


def eval_next_word_prediction(
        model, tokenizer, lm_dataset, accelerator=None,
        batch_size: int = 4,
        new_token_ids=None, replaced_token_seqs_by_len=None,
        new_token_to_original_first_token=None,
        max_length: int = 256,
        drop_last: bool = True,
        eval_max_samples: int = None,
        eval_shuffle_samples: bool = False,
        reduction="none",
        test_mode=False
):
    if accelerator is None:
        accelerator = Accelerator()
    model.eval()
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
    else:
        add_start_token = False

    data_collator = default_data_collator

    if eval_max_samples:
        eval_idx = range(len(lm_dataset), min(eval_max_samples, len(lm_dataset)))
        if eval_shuffle_samples:
            eval_idx = np.random.choice(len(lm_dataset), min(eval_max_samples, len(lm_dataset)))
        lm_dataset = lm_dataset.select(eval_idx)

    # Create data loaders
    eval_dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=drop_last, shuffle=False,
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)
    
    model.eval()


    bits_per_byte_evaluator = BitsPerByte(tokenizer, device=model.device)
    bits_per_byte_values_per_seq = torch.Tensor([]).to(model.device)

    if new_token_ids is not None:
        new_token_ids = torch.tensor(new_token_ids).to(model.device)
    if replaced_token_seqs_by_len is not None:
        # tensorize the sequences for efficient comparison
        replaced_token_seqs_by_len = {token_length: torch.tensor(skip_token_seqs).to(model.device) for token_length, skip_token_seqs in replaced_token_seqs_by_len.items() if len(skip_token_seqs) > 0}
    if new_token_to_original_first_token is not None:
        # Convert the mapping into a tensor for efficient indexing, create a mapping tensor that defaults to identity
        # All original tokens map to themselves; new tokens map to the first sub-word according to the original tokenization
        new_token_to_orig_first_mapping_tensor = torch.arange(len(tokenizer), device=model.device)
        new_token_to_orig_first_mapping_tensor[torch.tensor(list(new_token_to_original_first_token.keys()), device=model.device)] = \
            torch.tensor(list(new_token_to_original_first_token.values()), device=model.device)

    target_metrics = defaultdict(list)
    background_metrics = defaultdict(list)
    overall_metrics = defaultdict(list)

    # run eval and compute metrics
    for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), miniters=10, desc="Evaluating vocabulary..."):
        if add_start_token:
            batch = _add_start_token(batch, tokenizer)

        labels = batch["input_ids"]             #batch_size x seq_len
        attn_mask = batch["attention_mask"]     #batch_size x seq_len
        batch.pop("labels")
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits             #batch_size x seq_len x vocab_size

        # Take the logits for only the first seq_len -1 tokens since we don't 
        # have a label for the logit distribution
        shift_logits = out_logits[..., :-1, :].contiguous() #batch_size x (seq_len - 1) x vocab_size
        shift_labels = labels[..., 1:].contiguous()     #batch_size x (seq_len - 1)
        # Shift attention mask with the label so we know where we should not be 
        # evaluating the label (e.g., for [PAD] tokens)
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()    #batch_size x (seq_len - 1)

        
        # Compute bits per byte for the current batch and concat to bits_per_byte_values_per_seq tensor
        bits_per_byte_values_per_seq = torch.cat([bits_per_byte_values_per_seq, bits_per_byte_evaluator.compute_bits_per_byte(shift_logits, shift_labels, shift_attention_mask_batch)], dim=0)
        
        # shift_attention_mask_batch: (batch_size, seq_len) masks out all the new tokens 
        # added and the length of the original sequence (in the case of base tokenizer)
        shift_attention_mask_batch, ignore_mask = \
            _ignore_new_words_in_attention_mask(
                shift_attention_mask_batch, shift_labels, new_token_ids, replaced_token_seqs_by_len)
        # Contains first tokens of original tokenization for the new tokens added
        original_labels = None if new_token_to_original_first_token is None \
            else new_token_to_orig_first_mapping_tensor[shift_labels]
        # Contains logits for original tokens only (without new tokens added)
        original_logits = None if new_token_ids is None else torch.cat([shift_logits[:, :, :min(new_token_ids)], shift_logits[:, :, max(new_token_ids)+1:]], dim=-1)
        
        background_results, target_results, overall_results, successful_targets = \
            compute_metrics(
                shift_logits, shift_labels, shift_attention_mask_batch,
                original_labels=original_labels, original_logits=original_logits, compute_perplexity=True, testing_mode=test_mode)

        for metric_name, metric_value in target_results.items():
            target_metrics[metric_name].append(np.array(metric_value))
        for metric_name, metric_value in background_results.items():
            background_metrics[metric_name].append(metric_value)
        for metric_name, metric_value in overall_results.items():
            overall_metrics[metric_name].append(metric_value)

    eval_dataloader = accelerator.free_memory(eval_dataloader)

    def _concat_func(x):
        if isinstance(x, np.ndarray) and len(x.shape) > 1:
            x = np.concat(x)
        elif isinstance(x, (list, tuple)) and len(x) > 1:
            if isinstance(x[0], np.ndarray) and len(x[0].shape) == 0:
                x = np.array(x)
            else:
                x = np.concat(x)
        return x

    # apply reduction
    reduce_func = _concat_func
    if reduction == 'mean':
        reduce_func = lambda x: np.mean(_concat_func(x)).item()

    target_metrics["count"] = None
    for metric_name, metric_value in target_metrics.items():
        target_metrics[metric_name] = reduce_func(metric_value)
        if target_metrics["count"] is None:
            target_metrics["count"] = len(_concat_func(metric_value))
    for metric_name, metric_value in background_metrics.items():
        background_metrics[metric_name] = reduce_func(metric_value)
    for metric_name, metric_value in overall_metrics.items():
        overall_metrics[metric_name] = reduce_func(metric_value)

    
    # Total sequences evaluated in bits per byte
    print("Total sequences evaluated in bits per byte: ", len(bits_per_byte_values_per_seq))
    # Convert to numpy and average of bits per byte to overall metrics
    overall_metrics["bits_per_byte"] = bits_per_byte_values_per_seq.cpu().numpy().mean()
    # Average overall sequences 
    print("Average bits per byte: ", overall_metrics["bits_per_byte"])

    return background_metrics, target_metrics, overall_metrics



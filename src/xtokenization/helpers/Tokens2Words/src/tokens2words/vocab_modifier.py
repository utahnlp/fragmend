from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
import numpy as np
import pandas as pd
import re
import regex
import torch
from torch import nn
from collections import defaultdict
from typing import DefaultDict
import tempfile
import json
from copy import deepcopy

from .representation_translator import RepresentationTranslators
from .word_retriever import PatchscopesRetriever
from .utils.calibration_utils import get_calibration_model, train_calibration_model, merge_calibrators_to_hf_model
from .utils.model_utils import extract_token_i_hidden_states
from .utils.subword_utils import SubwordRepresentation, SubwordRepresentationStore

import logging

logger = logging.getLogger(__name__)

class VocabularyModifier(ABC):
    """
    Abstract class for...  # TODO
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            base_tokenizer: PreTrainedTokenizer = None,
            add_to_core_vocab: bool = False,
            add_space_before_lowercase_words: bool = False,
            add_space_before_all_words: bool = False,
            space_token: str = "Ġ",
            batch_size: int = 64,
            rep_pasting=False,
            random_emb=False,
            subword_emb=False,
            random_emb_norm=False,
            focus_emb=False,
            add_constituent_subwords=False,
            min_constituent_subword_length=None,
            subword_rep_reduction_strategy="mean",
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_tokenizer = base_tokenizer if base_tokenizer is not None else deepcopy(tokenizer)

        self.add_to_core_vocab = add_to_core_vocab
        self.add_space_before_lowercase_words = add_space_before_lowercase_words
        self.add_space_before_all_words = add_space_before_all_words
        self.space_token = space_token

        self.orig_vocab_size = len(tokenizer) if not self.add_to_core_vocab else len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))
        self.num_special_tokens = len(tokenizer._tokenizer.get_added_tokens_decoder())
        self.new_token_ids: List[int] = list()
        self.new_words: List[str] = list()
        self.failed_words: List[str] = list()
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}
        self.batch_size = batch_size

        self.calibrators = None
        self.rep_pasting = rep_pasting
        self.random_emb = random_emb
        self.subword_emb = subword_emb
        self.random_emb_norm = random_emb_norm
        self.focus_emb = focus_emb

        self.get_readable_vocab_list = self.get_readable_vocab_list()
        self.add_constituent_subwords = add_constituent_subwords
        self.min_constituent_subword_length = min_constituent_subword_length
        if self.min_constituent_subword_length is not None:
            self.min_constituent_subword_length = int(self.min_constituent_subword_length)
        self.subword_rep_reduction_strategy = subword_rep_reduction_strategy

    def get_readable_vocab_list(self):
        """ Since we are adding string tokens to the tokenizer, 
        we can get a list of the 'readable' tokens in the tokenizer's vocab.
        We will use this read to filter redundant tokens
        """
        vocab = self.tokenizer.get_vocab()
        readable_vocab = set()
        for _, id in vocab.items():
            try:
                decoded_token = self.tokenizer.decode([id])
                if decoded_token not in readable_vocab:
                    readable_vocab.add(decoded_token)
            except:
                pass
        return list(readable_vocab)

    @abstractmethod
    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the entries in the embedding and LM head matrices for a given word.

        Args:
            word (str): The word to add to the vocabulary

        Returns:
            embedding entry (torch.Tensor): The transformed representation in embedding space.
            lm_head entry (torch.Tensor): The transformed representation in LM head space.
        """
        pass

    @abstractmethod
    def compute_entries_for_words(
            self, word: str | List[str]
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the entries in the embedding and LM head matrices for a given word.

        Args:
            word (str): The word to add to the vocabulary

        Returns:
            embedding entry (torch.Tensor): The transformed representation in embedding space.
            lm_head entry (torch.Tensor): The transformed representation in LM head space.
        """
        pass

    @abstractmethod
    def compute_entries_for_words_and_subwords(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the entries in the embedding and LM head matrices for a given word.

        Args:
            word (str): The word to add to the vocabulary

        Returns:
            embedding entry (torch.Tensor): The transformed representation in embedding space.
            lm_head entry (torch.Tensor): The transformed representation in LM head space.
        """
        pass

    def undo_vocabulary_changes(self):
        self.tokenizer = deepcopy(self.base_tokenizer)
        
        self.orig_vocab_size = len(self.tokenizer) #if not self.add_to_core_vocab else len(self.tokenizer._tokenizer.get_vocab(with_added_tokens=False))
        self.num_special_tokens = len(self.tokenizer._tokenizer.get_added_tokens_decoder())
        self.new_token_ids = list()
        self.new_words = list()
        self.failed_words = list()
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}
        self.model.resize_token_embeddings(len(self.tokenizer))

        return self.model, self.tokenizer

    def add_words_to_core_vocab(self, words: List[str], token_ids: List[int]) -> None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer to the temporary directory
            self.tokenizer.save_pretrained(temp_dir)

            # Load the tokenizer.json file
            tokenizer_json_path = f"{temp_dir}/tokenizer.json"
            with open(tokenizer_json_path, 'r') as f:
                tokenizer_json = json.load(f)

            # Make some modifications to the tokenizer.json (example: add a custom entry)
            for word, token_id in zip(words, token_ids):
                tokenizer_json['model']['vocab'][word] = token_id

            # Save the modified tokenizer.json file
            with open(tokenizer_json_path, 'w') as f:
                json.dump(tokenizer_json, f, indent=2)

            # Reload the modified tokenizer (optional)
            self.tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    def add_word_to_vocab(
            self, word: str | List[str],
            finalize: bool = True,
            embedding_entry: torch.Tensor = None, lm_head_entry: torch.Tensor = None,
    ) -> bool:
        if embedding_entry is not None and lm_head_entry is not None:
            # use precomputed entry representations
            pass
        else:
            # MM Comment
            # embedding_entry, lm_head_entry = self.compute_entries_for_word(word)
            embedding_entry, lm_head_entry, failed_words = self.compute_entries_for_words(word)
            self.failed_words.extend(failed_words)
            if embedding_entry is None or lm_head_entry is None:
                return False

        if isinstance(word, str):
            word = [word]
        
        emb_ix = 0
        num_added_tokens = 0
        for ix, w in enumerate(word):
            if w in failed_words:
                continue

            if self.add_space_before_lowercase_words and w[0].islower():
                w = self.space_token + w
            elif self.add_space_before_all_words:
                w = self.space_token + w

            # Add new word to the tokenizer
            num_tokens_in_w = int(w not in self.tokenizer.get_vocab() and (len(self.tokenizer.tokenize(w)) != 1))
            num_added_tokens += 1

            if num_tokens_in_w > 0:  # don't add word if it already exists
                self.new_words.append(w)
                if finalize:
                    self.tokenizer.add_tokens([w])
                    self.model.resize_token_embeddings(len(self.tokenizer) + num_tokens_in_w, mean_resizing=False)
                    new_token_idx = len(self.tokenizer) - 1
                    if self.add_to_core_vocab:
                        new_token_idx -= self.num_special_tokens
                    self.new_token_ids.append(new_token_idx)

                    with torch.no_grad():
                        self.model.get_input_embeddings().weight[new_token_idx] = embedding_entry[emb_ix]
                        self.model.get_output_embeddings().weight[new_token_idx] = lm_head_entry[emb_ix]
                else:
                    self.entries_cache["embedding"][w] = embedding_entry[emb_ix]
                    self.entries_cache["lm_head"][w] = lm_head_entry[emb_ix]

            emb_ix += 1

        return num_added_tokens > 0



    def add_words_to_vocab(
            self, words: Iterable[str],
            precomputed_embedding_entries: Dict[str, torch.Tensor] = None,
            precomputed_lm_head_entries: Dict[str, torch.Tensor] = None,
    ):
        # Create a subword representation store to track representations of a subword occuring in multiple words
        
        if self.add_constituent_subwords:
            self.embedding_subword_repstore = SubwordRepresentationStore(curation_strategy=self.subword_rep_reduction_strategy)
            self.lm_head_subword_repstore = SubwordRepresentationStore(curation_strategy=self.subword_rep_reduction_strategy)
            if self.random_emb_norm or self.focus_emb:
                emb_means = torch.mean(self.model.get_input_embeddings().weight, dim=0, keepdim=True)
                emb_stds = torch.std(self.model.get_input_embeddings().weight, dim=0, keepdim=True)
                lm_head_means = torch.mean(self.model.get_output_embeddings().weight, dim=0, keepdim=True)
                lm_head_stds = torch.std(self.model.get_output_embeddings().weight, dim=0, keepdim=True)
                # Define a normal distribution with the computed mean and variance
                self.emb_distribution = torch.distributions.Normal(emb_means, emb_stds)
                self.lm_head_distribution = torch.distributions.Normal(lm_head_means, lm_head_stds)

        if precomputed_embedding_entries is not None and precomputed_lm_head_entries is not None:
            self.entries_cache["embedding"] = precomputed_embedding_entries
            self.entries_cache["lm_head"] = precomputed_lm_head_entries
        else:
            ## MM Comment
            # for word in tqdm(words, total=len(words), desc="Computing entry representations for words...", unit="word"):
            #     # Adds words to entries cache and the word to new_word if retrieval was successful
            #     self.add_word_to_vocab(word, finalize=False)
            for i in tqdm(range(0,len(words), self.batch_size), total=int(len(words)/self.batch_size)+1, desc="Computing entry representations for words...", unit="word"):
                # Adds words to entries cache and the word to new_word if retrieval was successful
                word_batch = words[i:i+self.batch_size]
                if self.add_constituent_subwords:
                    self.compute_entries_for_words_and_subwords(word_batch)
                else:
                    self.add_word_to_vocab(word_batch, finalize=False)
            
            # 'New words to be added' are nothing but keys in the representation store
            if self.add_constituent_subwords:
                assert self.embedding_subword_repstore.get_all_tokens() == self.lm_head_subword_repstore.get_all_tokens(), "Tokens in embedding and LM head subword representation stores do not match"
                self.new_words = self.embedding_subword_repstore.get_all_tokens()
      
        self.new_token_ids = list(range(self.orig_vocab_size, self.orig_vocab_size+len(self.new_words)))
        if self.add_to_core_vocab:
            self.add_words_to_core_vocab(self.new_words, self.new_token_ids)
        else:
            for word in self.new_words:
                self.tokenizer.add_tokens([word])
            

        self.model.resize_token_embeddings(len(self.base_tokenizer) + len(self.new_words))
        if self.add_to_core_vocab:
            pass  # TODO adjust for direct editing of tokenizer

        for new_token_idx, word in zip(
                self.new_token_ids,
                self.new_words,
            ):
            with torch.no_grad():
                if self.add_constituent_subwords:
                    embedding_entry = self.embedding_subword_repstore.get_word_representation(word)
                    lm_head_entry = self.lm_head_subword_repstore.get_word_representation(word)
                    self.model.get_input_embeddings().weight[new_token_idx] = embedding_entry
                    self.model.get_output_embeddings().weight[new_token_idx] = lm_head_entry
                else:
                    self.model.get_input_embeddings().weight[new_token_idx] = self.entries_cache["embedding"][word]
                    self.model.get_output_embeddings().weight[new_token_idx] = self.entries_cache["lm_head"][word]
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}
        if self.add_constituent_subwords:
            print(f"Added words: {self.new_words[:45]}")
            print(len(self.new_words))
            print(f"Number of words with multiple subword representations: {len(self.embedding_subword_repstore.get_tokens_with_multiple_reps())}")
            # exit()

        return self.model, self.tokenizer

    def get_new_tokens_to_replaced_token_seqs_map(self, remove_prefix_space=True):
        mapping = {token_id: self.base_tokenizer.encode(word, add_special_tokens=False)
                   for word, token_id in zip(self.new_words, self.new_token_ids)}
        if remove_prefix_space:
            mapping = {token_id: encoding[1:] if not bool(self.base_tokenizer.decode(encoding[0])) else encoding
                       for token_id, encoding in mapping.items()}
        return mapping

    def train_and_apply_calibrators_to_new_entries(self, dataset, save_dir=None, overwrite_cache=False, max_samples=None, lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=4, max_length=256, n_warmup_steps=0, clip_grad_norm=1.0, target_loss_weight=0.15, subsequent_loss_weight=0.15, mixed_precision=None, lora_r=None, learn_gate_activation=False,
            existing_tokens_to_calibrate=None,
            new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
            existing_tokens_for_similarity_loss=None,):
        self.calibrators = self.train_calibrators(dataset, save_dir, overwrite_cache, max_samples, lr, lr_schedule, num_epochs, batch_size, max_length, n_warmup_steps, clip_grad_norm, target_loss_weight, subsequent_loss_weight, mixed_precision, lora_r, learn_gate_activation,
            existing_tokens_to_calibrate=existing_tokens_to_calibrate if existing_tokens_to_calibrate is not None else self.existing_tokens_to_calibrate,
            new_tokens_to_orig_first_map=new_tokens_to_orig_first_map, soft_allow_orig_first_tokens=soft_allow_orig_first_tokens,
            existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,)
        self.apply_calibrators_to_new_entries()
        return self.model

    def load_calibrators(self, save_dir):
        calibration_model = get_calibration_model(
            self.model, 
            self.orig_vocab_size, 
            len(self.new_words)
            )
        calibrators_loaded = calibration_model.load_calibrators(save_dir, fail_ok=True)
        if calibrators_loaded:
            return calibration_model.get_calibrators()
        return None

    def train_calibrators(self, dataset, save_dir=None, overwrite_cache=False, max_samples=None, 
                          lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=4, 
                          max_length=256, n_warmup_steps=0, clip_grad_norm=1.0, 
                          target_loss_weight=0.15, subsequent_loss_weight=0.15, 
                          mixed_precision=None, lora_r=None, learn_gate_activation=False,
                          learn_per_token_bias=False, existing_tokens_to_calibrate=None,
                          new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
                          existing_tokens_for_similarity_loss=None):
        
        calibration_model = get_calibration_model(
            self.model, 
            self.orig_vocab_size, 
            len(self.new_words), 
            learn_gate_activation=learn_gate_activation,
            existing_tokens_to_calibrate=existing_tokens_to_calibrate,
            new_tokens_to_orig_first_map=new_tokens_to_orig_first_map, 
            soft_allow_orig_first_tokens=soft_allow_orig_first_tokens,
            )

        if (learn_per_token_bias or existing_tokens_to_calibrate is not None) and max_samples is not None:
            max_samples = max_samples // 2

        train_calibrators = True
        if save_dir is not None and not overwrite_cache:
            calibrators_loaded = calibration_model.load_calibrators(save_dir, fail_ok=True)
            train_calibrators = not calibrators_loaded
        if train_calibrators:
            calibration_model = train_calibration_model(
                calibration_model, 
                self.tokenizer, 
                dataset, 
                save_dir,
                max_samples=max_samples, 
                lr=lr, 
                lr_schedule=lr_schedule, 
                num_epochs=num_epochs, 
                batch_size=batch_size, 
                max_length=max_length,
                n_warmup_steps=n_warmup_steps, 
                clip_grad_norm=clip_grad_norm, 
                mixed_precision=mixed_precision,
                freeze_existing_tokens=existing_tokens_to_calibrate is not None,
                existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,)

            # 2-step training when adding per-token bias term, or fine-tuning existing tokens
            if learn_per_token_bias or existing_tokens_to_calibrate is not None:
                if learn_per_token_bias:
                    calibration_model.set_use_bias(True)
                calibration_model = train_calibration_model(
                    calibration_model,
                    self.tokenizer,
                    dataset,
                    save_dir,
                    max_samples=max_samples,
                    lr=lr,
                    lr_schedule=lr_schedule,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    max_length=max_length,
                    n_warmup_steps=n_warmup_steps,
                    clip_grad_norm=clip_grad_norm,
                    mixed_precision=mixed_precision,
                    existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,
                )

        calibrators = calibration_model.get_calibrators()
        return calibration_model, calibrators

    def set_calibrators(self, calibrators):
        self.calibrators = calibrators

    def apply_calibrators_to_new_entries(self, 
            new_tokens_start=None, 
            new_tokens_end=None, 
            embedding_calibrator=None, 
            lm_head_calibrator=None,
            existing_tokens_to_calibrate=None,
            existing_tokens_embedding_calibrator=None,
            existing_tokens_lm_head_calibrator=None):
        if self.calibrators is not None:
            self.model = merge_calibrators_to_hf_model(self.model, **self.calibrators)
        else:
            assert (new_tokens_start is not None) and \
                   ((embedding_calibrator is not None) or
                    (lm_head_calibrator is not None)), \
                "To apply calibrators, you must either train them first or pass calibrators as parameters"

            self.model = merge_calibrators_to_hf_model(
                self.model,
                new_tokens_start=new_tokens_start,
                new_tokens_end=new_tokens_end,
                embedding_calibrator=embedding_calibrator,
                lm_head_calibrator=lm_head_calibrator,
                existing_tokens_to_calibrate=existing_tokens_to_calibrate if existing_tokens_to_calibrate is not None else self.existing_tokens_to_calibrate,
                existing_tokens_embedding_calibrator=existing_tokens_embedding_calibrator,
                existing_tokens_lm_head_calibrator=existing_tokens_lm_head_calibrator
            )
        return self.model

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer


class DetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            patchscopes_retriever: PatchscopesRetriever,
            patchscopes_results: Union[np.ndarray, pd.DataFrame, Dict[str, Dict[int, str]]] = None,
            patchscopes_force_starts_with_word: bool = True,
            translators: RepresentationTranslators = None,
            detokenization_decision_rule: str = "first_id_layer",
            detokenization_decision_rule_E: str = None,
            max_valid_layer: int = None,
            early_exit_layer: int = None,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_decision_rule = detokenization_decision_rule
        self.detokenization_decision_rule_E = detokenization_decision_rule_E
        self.max_valid_layer = max_valid_layer
        self.early_exit_layer = early_exit_layer

        self.patchscopes_retriever = patchscopes_retriever
        self.patchscopes_results = patchscopes_results
        self.patchscopes_force_starts_with_word = patchscopes_force_starts_with_word
        if patchscopes_results is None:
            # create dict that maps new words (str) to a list of their patchscopes output per layer
            self.patchscopes_results: DefaultDict[str, List[str]] = defaultdict(list)

        self.translators = translators

    def _decide_detokenization_end_layer(self, word: str, patchscopes_results: Iterable[str], decision_rule=None, use_punct=True):
        decision_rule = self.detokenization_decision_rule if decision_rule is None else decision_rule
        patchscopes_results = np.array(patchscopes_results).astype(str)
        if self.early_exit_layer is not None:
            patchscopes_results = patchscopes_results[:self.early_exit_layer]

        # Check if each layer's result starts with the word
        # MM test
        patchscopes_results = np.char.strip(patchscopes_results)
        # print(patchscopes_results)
        starts_with_word = np.char.startswith(patchscopes_results, word)
        # print(starts_with_word)
        # print(word)
        # print(patchscopes_results[4])
        # Count occurrences of word in each layer's result
        # Use word boundary \b to match whole words only
        # or, word plus ',' or '.' to account for punctuation right after the word
        if use_punct:
            # Pattern must match for only the word, or the word followed by a comma or period,
            # and not for cases where the word is a substring of another word (e.g., "cat" should not match "caterpillar")
            pattern = f"\\b{re.escape(word)}\\b|\\b{re.escape(word)}(?=[,\\.])"
            pattern = f"\\b{re.escape(word)}\\b|\\b{re.escape(word)}(?=[,\\.])"
            pattern = rf'(?<!\w){re.escape(word)}(?!\w)\W?'
            pattern = rf'(?<![\p{{L}}\p{{M}}]){regex.escape(word)}(?![\p{{L}}\p{{M}}])\W?'
            counts = np.array([len(regex.findall(pattern, s)) for s in patchscopes_results])
        else:
            pattern = f"\\b{re.escape(word)}\\b"
            counts = np.array([len(re.findall(pattern, s)) for s in patchscopes_results])
        # print(counts)
        
        counts[~starts_with_word] = 0
        if np.all(counts == 0):
            return None
        
        result = None
        if decision_rule in ["first_id_layer", "1st_id_layer"]:
            result = np.argmax(counts > 0).item()
        if decision_rule in ["2nd_id_layer", "3rd_id_layer", "4th_id_layer", "4th_id_layer"]:
            indices = np.where(counts > 0)[0]
            if (decision_rule == "4th_id_layer") and (len(indices) >= 4):
                result = indices[3]
            elif (decision_rule in ["3rd_id_layer", "4th_id_layer"]) and (len(indices) >= 3):
                result = indices[2]
            elif len(indices) >= 2:
                result = indices[1]
            elif len(indices) >= 1:
                result = indices[0]
        if decision_rule == "max_id_layer":
            result = np.argmax(counts).item()
        elif decision_rule == "last_id_layer":
            result = (len(counts) - np.argmax((counts > 0)[::-1]) - 1).item()
        elif decision_rule == "first_layer_with_2_repeats":
            result = (np.argmax(counts >= 2)).item()
        elif decision_rule == "last_layer_with_2_repeats":
            result = (len(counts) - np.argmax((counts >= 2)[::-1]) - 1).item()

        if self.max_valid_layer is not None and result > self.max_valid_layer:
            # default to first id layer
            result = np.argmax(counts > 0).item()

        return result

    def compute_entries_for_word(
            self, word: str | List[str]
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        if word not in self.patchscopes_results:
            patchscopes_description_by_layers, last_token_hidden_states, _ = \
                self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word)
            self.patchscopes_results[word] = patchscopes_description_by_layers
        else:
            patchscopes_description_by_layers = self.patchscopes_results[word]
            last_token_hidden_states = self.patchscopes_retriever.extract_hidden_states(word)

        target_layer = target_layer_E = self._decide_detokenization_end_layer(word, patchscopes_description_by_layers)
        if self.detokenization_decision_rule_E is not None:
            target_layer_E = self._decide_detokenization_end_layer(
                word, patchscopes_description_by_layers, self.detokenization_decision_rule_E)

        
        if target_layer is None:  # detokenization did not occur
            return None, None

        target_as_embedding = last_token_hidden_states[target_layer_E]
        target_as_lm_head = last_token_hidden_states[target_layer]

        target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype)
        target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head
    
    ## MM Added
    def compute_entries_for_words(
            self, word: str | List[str]
    ) -> (torch.Tensor, torch.Tensor, list):
        """

        Args:
            word (str):
                ...
        """
        word_list_len = 1
        if isinstance(word, list):
            word_list_len = len(word)

        #if word not in self.patchscopes_results:
        patchscopes_description_by_layers, last_token_hidden_states, patchscopes_outputs = \
            self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word)
        last_token_hidden_states = last_token_hidden_states.reshape(word_list_len, int(last_token_hidden_states.shape[0]/word_list_len),last_token_hidden_states.shape[-1])
        num_layers =  last_token_hidden_states.shape[1]
        patchscopes_description_by_layers = [patchscopes_description_by_layers[i:i+num_layers] for i in range(0, len(patchscopes_description_by_layers), num_layers)]
        failed_words = []
        target_as_embedding = None
        target_as_lm_head = None

        # Compute mean and variance of embeddings and unembedddings
        if self.random_emb_norm:
            emb_means = torch.mean(self.model.get_input_embeddings().weight, dim=0, keepdim=True)
            emb_stds = torch.std(self.model.get_input_embeddings().weight, dim=0, keepdim=True)
            lm_head_means = torch.mean(self.model.get_output_embeddings().weight, dim=0, keepdim=True)
            lm_head_stds = torch.std(self.model.get_output_embeddings().weight, dim=0, keepdim=True)
            # Define a normal distribution with the computed mean and variance
            emb_distribution = torch.distributions.Normal(emb_means, emb_stds)
            lm_head_distribution = torch.distributions.Normal(lm_head_means, lm_head_stds)


        for ix, w in enumerate(word):
            self.patchscopes_results[w] = patchscopes_description_by_layers[ix]
        # else:
        #     patchscopes_description_by_layers = self.patchscopes_results[word]
        #     last_token_hidden_states = self.patchscopes_retriever.extract_hidden_states(word)
            target_layer = target_layer_E = self._decide_detokenization_end_layer(w, patchscopes_description_by_layers[ix])
            if self.detokenization_decision_rule_E is not None:
                target_layer_E = self._decide_detokenization_end_layer(
                    w, patchscopes_description_by_layers[ix], self.detokenization_decision_rule_E)
           
            if target_layer is None:  # detokenization did not occur
                failed_words.append(w)
                continue
            
            if self.rep_pasting:
                target_as_embedding_inst = last_token_hidden_states[ix,target_layer_E]
                # Patchscopes outputs hidden state are listed by generation_index x decoder layer x reps
                # Note that we have batch input for the patchscopes, hence, reps is of size (patchscopes_batch_size*num_layers) x generation_length x hidden_size
                try:
                    first_dim = ix*self.model.config.num_hidden_layers + target_layer_E
                except:
                    first_dim = ix*self.model.config.text_config.num_hidden_layers + target_layer_E
                target_as_lm_head_inst = patchscopes_outputs.hidden_states[0][-1][first_dim,-1,:]   
                
                if target_as_embedding is None:
                    target_as_embedding = target_as_embedding_inst.to(self.model.get_input_embeddings().weight.dtype).unsqueeze(0)
                    target_as_lm_head = target_as_lm_head_inst.to(self.model.get_output_embeddings().weight.dtype).unsqueeze(0)
                else:
                    target_as_embedding = torch.cat([target_as_embedding, target_as_embedding_inst.to(self.model.get_input_embeddings().weight.dtype).unsqueeze(0)], dim=0)
                    target_as_lm_head = torch.cat([target_as_lm_head, target_as_lm_head_inst.to(self.model.get_output_embeddings().weight.dtype).unsqueeze(0)], dim=0)
            elif (self.random_emb or self.random_emb_norm or self.subword_emb or self.focus_emb):
                if self.random_emb or self.focus_emb:
                    target_as_embedding_inst = torch.rand_like(last_token_hidden_states[ix,target_layer_E]).to(self.model.get_input_embeddings().weight.dtype).unsqueeze(0)
                    target_as_lm_head_inst = torch.rand_like(last_token_hidden_states[ix,target_layer]).to(self.model.get_output_embeddings().weight.dtype).unsqueeze(0)
                elif self.random_emb_norm:
                    target_as_embedding_inst = emb_distribution.sample().to(self.model.get_input_embeddings().weight.dtype)
                    target_as_lm_head_inst = lm_head_distribution.sample().to(self.model.get_output_embeddings().weight.dtype)
                    
                elif self.subword_emb:
                    sub_token_ids = self.base_tokenizer.encode(w, add_special_tokens=False)
                    target_as_embedding_inst = self.model.get_input_embeddings().weight[sub_token_ids].mean(dim=0, keepdim=True).to(self.model.get_input_embeddings().weight.dtype)
                    target_as_lm_head_inst = self.model.get_output_embeddings().weight[sub_token_ids].mean(dim=0, keepdim=True).to(self.model.get_output_embeddings().weight.dtype)

                if target_as_embedding is None:
                    target_as_embedding = target_as_embedding_inst
                    target_as_lm_head = target_as_lm_head_inst
                else:
                    target_as_embedding = torch.cat([target_as_embedding, target_as_embedding_inst], dim=0)
                    target_as_lm_head = torch.cat([target_as_lm_head, target_as_lm_head_inst], dim=0)
            else:
                target_as_embedding_inst = last_token_hidden_states[ix,target_layer_E]
                target_as_lm_head_inst = last_token_hidden_states[ix,target_layer_E]
                if target_as_embedding is None:
                    target_as_embedding = self.translators.to_embedding(target_as_embedding_inst.cpu(), target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype).unsqueeze(0)
                    target_as_lm_head = self.translators.to_lm_head(target_as_lm_head_inst.cpu(), target_layer+1).to(self.model.get_output_embeddings().weight.dtype).unsqueeze(0)
                else:
                    target_as_embedding = torch.cat([target_as_embedding, self.translators.to_embedding(target_as_embedding_inst.cpu(), target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype).unsqueeze(0)], dim=0)
                    target_as_lm_head = torch.cat([target_as_lm_head, self.translators.to_lm_head(target_as_lm_head_inst.cpu(), target_layer+1).to(self.model.get_output_embeddings().weight.dtype).unsqueeze(0)], dim=0)

        return target_as_embedding, target_as_lm_head, failed_words


    def get_prefixes(self, word):
        prefixes = []
        # Start from the  penultimate character to ensure at least a prefix of length 2, 
        # and go up to the full word
        for i in range(len(word)-2, -1, -1):
            # Only consider prefixes that are not in the tokenizer's vocabulary, 
            # and that meet the minimum length requirement if specified
            prefix_word = word[i:]
            if i == 0:
                if self.add_space_before_lowercase_words and word[0].islower():
                    prefix_word = self.space_token + word
                elif self.add_space_before_all_words:
                    prefix_word = self.space_token + word

            if self.min_constituent_subword_length is not None and len(prefix_word) < self.min_constituent_subword_length:
                continue
            
            if prefix_word not in self.base_tokenizer.get_vocab() and prefix_word not in self.get_readable_vocab_list:
                prefixes.append(prefix_word)
        return prefixes

    ## MM Added
    def subdetokenizate_word(self, word, offsets_ids, offset_char_mapping, patchscopes_decodings, hidden_states, num_layers):
        """ Method to check for subdetokenization in prefixes of the word, and store successful representations
        """
        word_success=False
        
        for ix in range(len(offsets_ids)):
            current_subword = word[:offsets_ids[ix][1]]
            is_final_token = (ix == len(offsets_ids)-1)
            
            # Curate a list of all prefixes for the current subword, and check if any of them were 
            # successfully detokenized by patchscopes. If so, add the successfully detokenized prefix 
            # to the vocabulary, along with its corresponding hidden state representation from patchscopes 
            # output.This allows us to add subword tokens to the vocabulary for cases where patchscopes 
            # was not able to successfully detokenize the entire word, but was able to successfully 
            # detokenize a prefix of the word.
            all_prefixes = self.get_prefixes(current_subword)
            # Select the subset of decodings and hidden states corresponding to the current subword 
            relevant_decodings = [patchscopes_decodings[layer_ix*len(offsets_ids) + ix] for layer_ix in range(num_layers)]
            relevant_states = hidden_states.reshape(num_layers, -1, hidden_states.shape[-1])[:,ix,:].squeeze(1)

            # The exception to the length rule (if enforced) is complete candidate word
            # This is to perfectly replicate the results  of the original patchscopes based 
            #  vocabulary expansion method for the full word
            if (current_subword == word) and \
                (word not in all_prefixes):
                if self.add_space_before_lowercase_words and word[0].islower():
                    all_prefixes.append(self.space_token + word)
                elif self.add_space_before_all_words:
                    all_prefixes.append(self.space_token + word)
                else:
                    all_prefixes.append(current_subword)

            assert len(relevant_decodings) == relevant_states.shape[0]
            # print(f"Current subword: {current_subword}")
            for prefix_ix, prefix in enumerate(all_prefixes):
                #if ix == len(offsets_ids)-1 and current_subword == word and prefix==word:
                prefix_layer = self._decide_detokenization_end_layer(prefix.strip(), relevant_decodings, use_punct=True)
                # print(f"Prefix layer for {prefix} is: {prefix_layer}")
                is_full_word = False
                if is_final_token and prefix.strip() == word and prefix_layer is not None:
                    is_full_word = True

                if prefix_layer:
                    if is_final_token and prefix.strip() == word:
                        word_success=True
                    prefix_token = prefix
                    is_starting = False
                    # Check if space was added in front and if the prefix is the beginning of the word 
                    if prefix_ix == len(all_prefixes)-1 and prefix_token[0] in [word[0],self.space_token]:
                        is_starting = True
                    # Add the detokenized representation to the subword representation store 
                    # to be added to the vocabulary later
                    if self.focus_emb or self.random_emb_norm:
                        if self.embedding_subword_repstore.is_present_in_store(prefix_token):
                            continue
                        self.embedding_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.emb_distribution.sample().to(self.model.get_input_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))
                        self.lm_head_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.lm_head_distribution.sample().to(self.model.get_input_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))

                    elif self.subword_emb:
                        if self.embedding_subword_repstore.is_present_in_store(prefix_token):
                            continue
                        sub_token_ids = self.base_tokenizer.encode(prefix_token, add_special_tokens=False)
                        self.embedding_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.model.get_input_embeddings().weight[sub_token_ids].mean(dim=0, keepdim=True).to(self.model.get_input_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))
                        self.lm_head_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.model.get_output_embeddings().weight[sub_token_ids].mean(dim=0, keepdim=True).to(self.model.get_output_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))
                    else:
                        self.embedding_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.translators.to_embedding(
                                    relevant_states[prefix_layer].cpu(), 
                                    prefix_layer+1
                                    ).to(self.model.get_input_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))
                        self.lm_head_subword_repstore.add_subword_representation(
                            SubwordRepresentation(
                                token=prefix_token,
                                representation=self.translators.to_lm_head(
                                    relevant_states[prefix_layer].cpu(), 
                                    prefix_layer+1
                                    ).to(self.model.get_input_embeddings().weight.dtype),
                                layer=prefix_layer,
                                is_starting=is_starting,
                                full_word=is_full_word
                            ))
            
        if not word_success:
            self.failed_words.append(word)


    ## MM Added
    def compute_entries_for_words_and_subwords(
            self, word: str | List[str]
    ) -> None:
        """

        Args:
            word (str):
                ...
        """
        word_list_len = 1
        if isinstance(word, list):
            word_list_len = len(word)
        
        # Number of model layers
        model_layers = self.model.config.num_hidden_layers if hasattr(self.model.config, 'num_hidden_layers') else self.model.config.text_config.num_hidden_layers
        patchscopes_description_by_layers, last_token_hidden_states, patchscopes_outputs = \
            self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word, decode_batch_size=model_layers*2)
        
        # Extarct the relevant portion of the patchscopes output corresponding to each word in the batch
        try:
            num_layers = self.model.config.num_hidden_layers
        except AttributeError:
            num_layers = self.model.config.text_config.num_hidden_layers
        ids = self.tokenizer(word, return_tensors="pt", padding=True, return_attention_mask=True, return_offsets_mapping=True)
        # get number of subword tokens for each word in the batch 
        original_token_lengths = ids.attention_mask.sum(dim=1).tolist()
        start_ix = 0

        # Discriminate indices that belong to different words in the batch based on the original token lengths, and split the patchscopes description and hidden states accordingly
        for w_ix, w in enumerate(word):
            # For every words, we will have num_layers*num_subwords descriptions and hidden states in the patchscopes output
            end_ix = start_ix + num_layers*original_token_lengths[w_ix]
      
            # Remove the padding tokens from the offsets mapping, and get the corresponding text spans for the subword 
            # tokens that make up the word in the batch
            text_mapping = [w[start:end] for start, end in ids["offset_mapping"][w_ix][:original_token_lengths[w_ix]]]
            
            self.subdetokenizate_word(
                word=w, 
                offsets_ids=ids["offset_mapping"][w_ix][:original_token_lengths[w_ix]],
                offset_char_mapping=text_mapping, 
                patchscopes_decodings=patchscopes_description_by_layers[start_ix:end_ix],
                hidden_states=last_token_hidden_states[start_ix:end_ix],
                num_layers=num_layers
            )
            start_ix = end_ix



    

    def get_patchscopes_results(self):
        return pd.DataFrame.from_records(self.patchscopes_results)


class HeuristicDetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            translators: RepresentationTranslators = None,
            detokenization_layer: int = 5,
            embedding_detokenization_layer: int = None,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_layer = detokenization_layer
        self.embedding_detokenization_layer = embedding_detokenization_layer if embedding_detokenization_layer is not None else detokenization_layer
        self.translators = translators

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        last_token_hidden_states = extract_token_i_hidden_states(
            self.model, self.tokenizer, word, token_idx_to_extract=-1,
            return_dict=False, verbose=False)

        target_layer = self.detokenization_layer
        target_layer_E = self.embedding_detokenization_layer

        target_as_embedding = last_token_hidden_states[target_layer_E]
        target_as_lm_head = last_token_hidden_states[target_layer]

        target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).detach().to(self.model.get_input_embeddings().weight.dtype)
        target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).detach().to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head

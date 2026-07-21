import torch
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

from .utils.enums import MultiTokenKind, RetrievalTechniques
from .processor import RetrievalProcessor
from .utils.logit_lens import ReverseLogitLens
from .utils.model_utils import extract_token_i_hidden_states, extract_token_i_hidden_states_batch

logger = logging.getLogger(__name__)

class WordRetrieverBase(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=3):
        pass


class PatchscopesRetriever(WordRetrieverBase):
    def __init__(
            self,
            model,
            tokenizer,
            representation_prompt: str = "{word}",
            patchscopes_prompt: str = "Next is the same word twice: 1) {word} 2)",
            prompt_target_placeholder: str = "{word}",
            representation_token_idx_to_extract: int|None = -1,
            num_tokens_to_generate: int = 10,
            sample_top_k = False,
            num_returned_sequences=10,
    ):
        super().__init__(model, tokenizer)
        self.prompt_input_ids, self.prompt_target_idx = \
            self._build_prompt_input_ids_template(patchscopes_prompt, prompt_target_placeholder)
        self._prepare_representation_prompt = \
            self._build_representation_prompt_func(representation_prompt, prompt_target_placeholder)
        self.representation_token_idx = representation_token_idx_to_extract
        self.num_tokens_to_generate = num_tokens_to_generate
        self.sample_top_k = sample_top_k
        self.num_returned_sequences = num_returned_sequences

    def _build_prompt_input_ids_template(self, prompt, target_placeholder):
        prompt_input_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        target_idx = []

        if prompt:
            assert target_placeholder is not None, \
                "Trying to set a prompt for Patchscopes without defining the prompt's target placeholder string, e.g., [MASK]"

            prompt_parts = prompt.split(target_placeholder)
            #print(f"Prompt parts: {prompt_parts}")
            for part_i, prompt_part in enumerate(prompt_parts):
                #print(f"Prompt part: {prompt_part}")
                prompt_input_ids += self.tokenizer.encode(prompt_part, add_special_tokens=False)
                if part_i < len(prompt_parts)-1:
                    ### Adds zeros at the as palceholders in the input
                    # and target_idx shows where the target placeholder
                    # is in the input
                    target_idx += [len(prompt_input_ids)]
                    prompt_input_ids += [0]
        else:
            prompt_input_ids += [0]
            target_idx = [len(prompt_input_ids)]
        
        prompt_input_ids = torch.tensor(prompt_input_ids, dtype=torch.long)
        target_idx = torch.tensor(target_idx, dtype=torch.long)
        return prompt_input_ids, target_idx

    def _build_representation_prompt_func(self, prompt, target_placeholder):
        return lambda word: prompt.replace(target_placeholder, word)

    def generate_states(self, tokenizer, word='Wakanda', with_prompt=True):
        prompt = self.generate_prompt() if with_prompt else word
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        return input_ids

    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=None, batch_size=None):
        self.model.eval()

        # insert hidden states into patchscopes prompt
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        #print(self.prompt_input_ids)
        # This contains the iput IDs for the templated prompt which as '0's as placeholders
        # 1 x num_tokens_in_template x hidden_size
        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device)).unsqueeze(0)
        # Broadcasts the above tensor to number of layers of the model
        # num_layers x num_tokens_in_template x hidden_size
        # Note that when we use this the 'num_layers' dimenstions is the batch size
        batched_patchscope_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        
        # Repalces the target embeddings with the hidden state computed for the target token
        batched_patchscope_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.model.device)
        # Attention Mask
        # num_layers x num_tokens_in_template
        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.model.device)
        

        num_tokens_to_generate = num_tokens_to_generate if num_tokens_to_generate else self.num_tokens_to_generate

        if batch_size is None:
            with torch.no_grad():
                if not self.sample_top_k:
                    patchscope_outputs = self.model.generate(
                        do_sample=False, num_beams=1, top_p=1.0, temperature=None,
                        inputs_embeds=batched_patchscope_inputs, attention_mask=attention_mask,
                        max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id, 
                        return_dict_in_generate=True, output_hidden_states=True)
                else:
                    patchscope_outputs = self.model.generate(
                        do_sample=True, num_beams=self.num_returned_sequences, top_p=1.0, temperature=None,
                        inputs_embeds=batched_patchscope_inputs, attention_mask=attention_mask,
                        max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=self.num_returned_sequences, return_dict_in_generate=True, output_hidden_states=True)
                    #patchscope_outputs = patchscope_outputs.reshape(int(patchscope_outputs.shape[0]/self.num_returned_sequences),self.num_returned_sequences, patchscope_outputs.shape[-1])
        else:
            # If batch_size is provided, we need to generate in batches to avoid memory issues
            all_outputs = []
            for i in range(0, len(hidden_states), batch_size):
                batch_inputs_embeds = batched_patchscope_inputs[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                with torch.no_grad():
                    if not self.sample_top_k:
                        batch_outputs = self.model.generate(
                            do_sample=False, num_beams=1, top_p=1.0, temperature=None,
                            inputs_embeds=batch_inputs_embeds, attention_mask=batch_attention_mask,
                            max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id, 
                            return_dict_in_generate=True, output_hidden_states=True)
                    else:
                        batch_outputs = self.model.generate(
                            do_sample=True, num_beams=self.num_returned_sequences, top_p=1.0, temperature=None,
                            inputs_embeds=batch_inputs_embeds, attention_mask=batch_attention_mask,
                            max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=self.num_returned_sequences, return_dict_in_generate=True, output_hidden_states=True)
                all_outputs.append(batch_outputs)
            # Concatenate all outputs into a single ModelOutput class
            # patchscope_outputs = type(all_outputs[0])(
            #     sequences=torch.cat([o.sequences for o in all_outputs], dim=0),
            #     hidden_states=tuple(
            #         tuple(
            #             torch.cat([o.hidden_states[step][layer] for o in all_outputs], dim=0)
            #             for layer in range(len(all_outputs[0].hidden_states[0]))
            #         )
            #         for step in range(len(all_outputs[0].hidden_states))
            #     ),
            #     scores=tuple(
            #         torch.cat([o.scores[step] for o in all_outputs], dim=0)
            #         for step in range(len(all_outputs[0].scores))
            #     ) if all_outputs[0].scores is not None else None,
            # )
            # For space-saving we'll use just the sequences for now
            patchscope_outputs = type(all_outputs[0])(
                sequences=torch.cat([o.sequences for o in all_outputs], dim=0),
                hidden_states=None,
                scores=None,
            )

        decoded_patchscope_outputs = self.tokenizer.batch_decode(patchscope_outputs.sequences)
        
        return decoded_patchscope_outputs, patchscope_outputs



    def extract_hidden_states(self, word):
        # MM added first if condition
        word_list = False
        if type(word) is list:
            word_list = True
            representation_input = [self._prepare_representation_prompt(w) for w in word]
        else:
            representation_input = self._prepare_representation_prompt(word)
        
        if word_list:
            last_token_hidden_states = extract_token_i_hidden_states_batch(
                self.model, self.tokenizer, representation_input, token_idx_to_extract=self.representation_token_idx, return_dict=False, verbose=False, batch_size=len(representation_input))
            
            # last_token_hidden_states = last_token_hidden_states.reshape(-1, last_token_hidden_states.shape[-1])    
        else:
            last_token_hidden_states = extract_token_i_hidden_states(
                self.model, self.tokenizer, representation_input, token_idx_to_extract=self.representation_token_idx, return_dict=False, verbose=False)
            # The hidden states are of shape (batch_size, seq_len, hidden_size), make it (batch_size*seq_len, hidden_size)
            if last_token_hidden_states.dim() >= 3:
                last_token_hidden_states = last_token_hidden_states.reshape(-1, last_token_hidden_states.shape[-1])

        return last_token_hidden_states

    def get_hidden_states_and_retrieve_word(self, word, num_tokens_to_generate=None, decode_batch_size=None):
        last_token_hidden_states = self.extract_hidden_states(word)
        
        # First split by batch_size, then by layer, then by seq position and hidden size 
        patchscopes_description_by_layers, patchscopes_output = self.retrieve_word(
            last_token_hidden_states, num_tokens_to_generate=num_tokens_to_generate, batch_size=decode_batch_size)
        return patchscopes_description_by_layers, last_token_hidden_states, patchscopes_output


class ReverseLogitLensRetriever(WordRetrieverBase):
    def __init__(self, model, tokenizer, device='cuda', dtype=torch.float16):
        super().__init__(model, tokenizer)
        self.reverse_logit_lens = ReverseLogitLens.from_model(model).to(device).to(dtype)

    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=3):
        result = self.reverse_logit_lens(hidden_states, layer_idx)
        token = self.tokenizer.decode(torch.argmax(result, dim=-1).item())
        return token


class AnalysisWordRetriever:
    def __init__(self, model, tokenizer, multi_token_kind, num_tokens_to_generate=1, add_context=True,
                 model_name='LLaMa-2B', device='cuda', dataset=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.multi_token_kind = multi_token_kind
        self.num_tokens_to_generate = num_tokens_to_generate
        self.add_context = add_context
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.retriever = self._initialize_retriever()
        self.RetrievalTechniques = (RetrievalTechniques.Patchscopes if self.multi_token_kind == MultiTokenKind.Natural
                                    else RetrievalTechniques.ReverseLogitLens)
        self.whitespace_token = 'Ġ' if model_name in ['gemma-2-9b', 'pythia-6.9b', 'LLaMA3-8B', 'Yi-6B'] else '▁'
        self.processor = RetrievalProcessor(self.model, self.tokenizer, self.multi_token_kind,
                                            self.num_tokens_to_generate, self.add_context, self.model_name,
                                            self.whitespace_token)

    def _initialize_retriever(self):
        if self.multi_token_kind == MultiTokenKind.Natural:
            return PatchscopesRetriever(self.model, self.tokenizer)
        else:
            return ReverseLogitLensRetriever(self.model, self.tokenizer)

    def retrieve_words_in_dataset(self, number_of_examples_to_retrieve=2, max_length=1000):
        self.model.eval()
        results = []

        for text in tqdm(self.dataset['train']['text'][:number_of_examples_to_retrieve], self.model_name):
            tokenized_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(
                self.device)
            tokens = tokenized_input.input_ids[0]
            #print(f'Processing text: {text}')
            i = 5
            while i < len(tokens):
                if self.multi_token_kind == MultiTokenKind.Natural:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_word(
                        tokens, i, device=self.device)
                elif self.multi_token_kind == MultiTokenKind.Typo:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_typo(
                        tokens, i, device=self.device)
                else:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_separated(
                        tokens, i, device=self.device)

                if len(word_tokens) > 1:
                    with torch.no_grad():
                        outputs = self.model(**tokenized_combined_text, output_hidden_states=True)

                    hidden_states = outputs.hidden_states
                    for layer_idx, hidden_state in enumerate(hidden_states):
                        postfix_hidden_state = hidden_states[layer_idx][0, -1, :].unsqueeze(0)
                        retrieved_word_str = self.retriever.retrieve_word(postfix_hidden_state, layer_idx=layer_idx,
                                                                          num_tokens_to_generate=len(word_tokens))
                        results.append({
                            'text': combined_text,
                            'original_word': original_word,
                            'word': word,
                            'word_tokens': self.tokenizer.convert_ids_to_tokens(word_tokens),
                            'num_tokens': len(word_tokens),
                            'layer': layer_idx,
                            'retrieved_word_str': retrieved_word_str,
                            'context': "With Context" if self.add_context else "Without Context"
                        })
                else:
                    i = j
        return results

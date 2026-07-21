import torch
import numpy as np
from collections import OrderedDict

class SubwordRepresentation:
    """
    A class to represent a subword token and its associated information.
    """

    def __init__(self, token: str, is_starting: bool, full_word: bool, layer:int, representation: torch.Tensor):
        self.token = token
        self.is_starting = is_starting
        self.full_word = full_word
        self.layer = layer
        self.representation = representation



class SubwordRepresentationStore:
    """
    A class to manage a list of SubwordRepresentation objects.
    """

    def __init__(self, curation_strategy: str = "mean"):
        """Initialize the SubwordRepresentationStore with a blank dictionary to hold the subword representations.
        Args:
            curation_strategy (str, optional): The strategy to pool different representations. Defaults to "mean".
        """
        self.curation_strategy = curation_strategy
        self.subword_representations = OrderedDict()

    def add_subword_representation(self, subword_representation: SubwordRepresentation):
        if subword_representation.token in self.subword_representations:
            self.subword_representations[subword_representation.token].append(subword_representation)
        else:
            self.subword_representations[subword_representation.token] = [subword_representation]

    
    def get_word_representation(self, word: str) -> torch.Tensor:
        """Get the representation of a word by pooling the representations of its subwords.
        Args:
            word (str): The word for which to get the representation.
        Returns:
            torch.Tensor: The pooled representation of the word.
        """
        if word not in self.subword_representations:
            raise ValueError(f"No subword representations found for word: {word}")
        
        subword_reps = [rep.representation for rep in self.subword_representations[word]]

        curation_strategy = self.curation_strategy.lower()

        if "fwpref" in curation_strategy:
            curation_strategy = curation_strategy.split("_")[1]
            # Return the representation corresponding to the full word if it exists, 
            # otherwise fall back to the specified strategy
            for rep in self.subword_representations[word]:
                if rep.full_word:
                    return rep.representation

        if curation_strategy == "mean":
            return torch.mean(torch.stack(subword_reps), dim=0)
        elif curation_strategy == "earliest":
            # Return the representation corresponding to the earliest layer where
            # detokenization occurs
            # Get index of the earliest layer where detokenization occurs
            # We do not consider tie breaking
            earliest_layer = np.argmin(np.array([rep.layer for rep in self.subword_representations[word]]))
            return subword_reps[earliest_layer]


    def get_all_tokens(self):
        """Get all the tokens in the store.
        Returns:
            list: A list of all tokens in the store.
        """
        return list(self.subword_representations.keys())
    

    def is_present_in_store(self, word: str) -> bool:
        """Check if a word is present in the store.
        Args:
            word (str): The word to check for presence in the store.
        Returns:
            bool: True if the word is present in the store, False otherwise.
        """
        return word in self.subword_representations
    
    def get_tokens_with_multiple_reps(self,) -> list:
        """Get a list of tokens that have multiple representations in the store.
        Returns:
            list: A list of tokens that have multiple representations in the store.
        """
        return [token for token, reps in self.subword_representations.items() if len(reps) > 1]

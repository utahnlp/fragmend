import torch
from torch.nn import functional as F

from transformers import AutoTokenizer

torch.manual_seed(42)

class BitsPerByte():
    def __init__(self, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.byte_len_table = torch.tensor([len(t.encode("utf-8")) for t, i in sorted(self.vocab.items(), key=lambda x: x[1])], device=device)
    
    def compute_bits_per_byte(self, logits, labels, attention_mask):
        # Compute cross_entropy at every token position and then average it over the dataset
        cross_entropy_per_token = F.cross_entropy(
                    logits.transpose(1, 2), 
                    labels, 
                reduction="none"
                ) * attention_mask

        bytes_per_token = self.byte_len_table[labels]    # contains the number of bytes for each token in labels
        
        # Compute the total number of bytes
        total_bytes = (bytes_per_token * attention_mask).sum(1)
        
        # Multiply cross entropy per token by the number of bytes per token
        cross_entropy_per_byte = cross_entropy_per_token * bytes_per_token
        
        # Compute the total cross entropy
        total_cross_entropy = (cross_entropy_per_byte * attention_mask).sum(1)
        
        # Compute the bits per byte, normalize with ln(2) to convert to bits
        bits_per_byte = total_cross_entropy / (total_bytes * torch.log(torch.tensor(2.0)))

        return bits_per_byte




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    bits_per_byte = BitsPerByte(tokenizer)
    # Create dummy logits, labels and attention_masks
    # Batch size 2
    logits = torch.randn(2, 10, tokenizer.vocab_size)
    labels = torch.randint(0, tokenizer.vocab_size, (2, 10))
    attention_mask = torch.ones((2, 10), dtype=torch.long)
    # Set attention mask of second sequence and last three tokens to zero
    attention_mask[1, -3:] = 0
    
    print(bits_per_byte.compute_bits_per_byte(logits, labels, attention_mask))
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(
        self, text: str, tokenizer, window_length: int, stride: int
    ):
        token_ids = tokenizer.encode(text)
        
        self.input_chunks = []
        self.target_chunks = []
        for start in range(0, len(token_ids) - window_length, stride):
            end = start + window_length
            
            input_chunk = token_ids[start:end]
            target_chunk = token_ids[start + 1:end + 1]
            
            self.input_chunks.append(torch.tensor(input_chunk))
            self.target_chunks.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_chunks)

    def __getitem__(self, idx):
        return self.input_chunks[idx], self.target_chunks[idx]


def create_dataloader_v1(
    text: str, window_length=256, stride=128, 
    batch_size=4, shuffle=True, drop_last=True, num_workers=0    
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, window_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
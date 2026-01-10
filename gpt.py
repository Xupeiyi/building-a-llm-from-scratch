from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        num_heads, 
        embedding_dim, 
        context_dim, 
        context_length, 
        dropout_rate, 
        qkv_bias=False
    ):
        super().__init__()
        assert (
            context_dim % num_heads == 0
        ), "context dimension must be divisible by number of heads"


        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = context_dim // num_heads

        self.W_query = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.W_key = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.W_value = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length), 
                diagonal=1
            )
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.out_projection = nn.Linear(context_dim, context_dim)
    
    def forward(self, embeddings):
        num_batches, context_length, embedding_dim = embeddings.shape

        keys = self.W_key(embeddings)
        queries = self.W_query(embeddings)
        values = self.W_value(embeddings)

        keys = keys.view(num_batches, context_length, self.num_heads, self.head_dim)
        queries = queries.view(num_batches, context_length, self.num_heads, self.head_dim)
        values = values.view(num_batches, context_length, self.num_heads, self.head_dim)

        # transpose to (num_batches, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:context_length, :context_length]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / (self.head_dim**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.view(num_batches, context_length, self.context_dim)

        context_vectors = self.out_projection(context_vectors)
        return context_vectors


class LayerNorm(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift
    

class GELU(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * torch.pow(x, 3))))
    

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):

    def __init__(
        self, 
        num_heads, 
        embedding_dim, 
        context_length,
        dropout_rate,
        qkv_bias,
    ):
        super().__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            context_dim=embedding_dim,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)
        self.drop_shortcut = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):

    def __init__(
        self, 
        vocabulary_size, 
        embedding_dim, 
        context_length, 
        num_heads,
        qkv_bias,
        num_transformers,
        dropout_rate
    ):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding_layer = nn.Embedding(context_length, embedding_dim)
        self.embedding_dropout_layer = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    context_length=context_length,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias
                ) 
                for _ in range(num_transformers)
            ]
        )
        self.final_norm = LayerNorm(embedding_dim)
        self.out_head = nn.Linear(embedding_dim, vocabulary_size, bias=False)

    def forward(self, token_ids):
        num_batches, context_length = token_ids.shape

        token_embeddings = self.token_embedding_layer(token_ids)
        position_embeddings = self.position_embedding_layer(
            torch.arange(context_length, device=token_ids.device)
        )
        embeddings = token_embeddings + position_embeddings
        embeddings = self.embedding_dropout_layer(embeddings)

        context_vectors = self.transformer_blocks(embeddings)
        context_vectors = self.final_norm(context_vectors)

        logits = self.out_head(context_vectors)
        return logits


def generate_text_simple(model, token_ids, num_new_tokens: int, context_size: int):
    # token_ids is a (num_batches, num_tokens) array of token ids in the current context

    for _ in range(num_new_tokens):
        token_ids_in_context = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(token_ids_in_context)
        
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token_ids), dim=1)
    
    return token_ids


@dataclass
class GPTConfig:
    vocabulary_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_transformers: int
    dropout_rate: float
    qkv_bias: bool


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
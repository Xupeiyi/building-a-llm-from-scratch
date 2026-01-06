import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        n_heads, 
        embedding_dim, 
        context_dim, 
        context_length, 
        dropout_rate, 
        qkv_bias=False
    ):
        super().__init__()
        assert (
            context_dim % n_heads == 0
        ), "context dimension must be divisible by number of heads"


        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = context_dim // n_heads

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
        n_batches, context_length, embedding_dim = embeddings.shape

        keys = self.W_key(embeddings)
        queries = self.W_query(embeddings)
        values = self.W_value(embeddings)

        keys = keys.view(n_batches, context_length, self.n_heads, self.head_dim)
        queries = queries.view(n_batches, context_length, self.n_heads, self.head_dim)
        values = values.view(n_batches, context_length, self.n_heads, self.head_dim)

        # transpose to (n_batches, n_heads, n_tokens, head_dim) 
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
        context_vectors = context_vectors.view(n_batches, context_length, self.context_dim)
        
        context_vectors = self.out_projection(context_vectors)
        return context_vectors

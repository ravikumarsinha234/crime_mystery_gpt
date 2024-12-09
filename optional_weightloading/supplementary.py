import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_sequences = []
        self.target_sequences = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for start_idx in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[start_idx:start_idx + max_length]
            target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
            self.input_sequences.append(torch.tensor(input_chunk))
            self.target_sequences.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.target_sequences[index]


def create_dataloader_v1(text, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout_rate, num_heads, use_qkv_bias=False):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads  # Projection dim per head

        self.query_projection = nn.Linear(input_dim, output_dim, bias=use_qkv_bias)
        self.key_projection = nn.Linear(input_dim, output_dim, bias=use_qkv_bias)
        self.value_projection = nn.Linear(input_dim, output_dim, bias=use_qkv_bias)
        self.output_projection = nn.Linear(output_dim, output_dim)  # Combine head outputs
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('causal_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_tensor):
        batch_size, num_tokens, input_dim = input_tensor.shape

        keys = self.key_projection(input_tensor)
        queries = self.query_projection(input_tensor)
        values = self.value_projection(input_tensor)

        # Split into heads and transpose dimensions for dot product attention
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention with a causal mask
        attention_scores = queries @ keys.transpose(2, 3)
        causal_mask = self.causal_mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(causal_mask, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute context vector and combine heads
        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.output_dim)
        output_tensor = self.output_projection(context_vector)

        return output_tensor


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        normalized_tensor = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        return self.scale * normalized_tensor + self.shift


class GELU(nn.Module):
    def forward(self, input_tensor):
        return 0.5 * input_tensor * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (input_tensor + 0.044715 * torch.pow(input_tensor, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            GELU(),
            nn.Linear(4 * config["embedding_dim"], config["embedding_dim"]),
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim=config["embedding_dim"],
            output_dim=config["embedding_dim"],
            context_length=config["context_length"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"],
            use_qkv_bias=config["qkv_bias"])
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config["embedding_dim"])
        self.norm2 = LayerNorm(config["embedding_dim"])
        self.shortcut_dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input_tensor):
        shortcut = input_tensor
        normalized_tensor = self.norm1(input_tensor)
        attention_output = self.attention(normalized_tensor)
        attention_output = self.shortcut_dropout(attention_output)
        input_tensor = attention_output + shortcut

        shortcut = input_tensor
        normalized_tensor = self.norm2(input_tensor)
        feed_forward_output = self.feed_forward(normalized_tensor)
        feed_forward_output = self.shortcut_dropout(feed_forward_output)
        return feed_forward_output + shortcut


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.embedding_dropout = nn.Dropout(config["dropout_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_layers"])])

        self.final_layer_norm = LayerNorm(config["embedding_dim"])
        self.output_layer = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)

    def forward(self, input_indices):
        batch_size, sequence_length = input_indices.shape
        token_embeddings = self.token_embedding(input_indices)
        position_embeddings = self.position_embedding(torch.arange(sequence_length, device=input_indices.device))
        input_tensor = token_embeddings + position_embeddings
        input_tensor = self.embedding_dropout(input_tensor)
        input_tensor = self.transformer_blocks(input_tensor)
        input_tensor = self.final_layer_norm(input_tensor)
        return self.output_layer(input_tensor)

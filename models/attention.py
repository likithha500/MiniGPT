
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention for a decoder-only GPT.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.block_size = block_size

        # Learnable Query, Key and Value projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Final projection after combining attention heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Lower triangular causal mask
        mask = torch.tril(
            torch.ones(block_size, block_size)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, block_size, block_size)
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        if sequence_length > self.block_size:
            raise ValueError(
                "Sequence length cannot exceed block_size"
            )

        # Generate Query, Key and Value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split embedding dimensions into attention heads
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Calculate scaled dot-product attention scores
        scores = q @ k.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # Apply causal masking
        mask = self.causal_mask[
            :, :, :sequence_length, :sequence_length
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=-1)

        weights = self.attention_dropout(weights)

        # Combine information from Value vectors
        output = weights @ v

        # Join attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        # Final output projection
        output = self.output_projection(output)

        output = self.residual_dropout(output)

        return output

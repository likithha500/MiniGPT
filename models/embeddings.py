
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into learnable embedding vectors.
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Provides learnable positional information to token sequences.
    """

    def __init__(self, block_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        return self.embedding(positions)

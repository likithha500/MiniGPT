
import torch
import torch.nn as nn

from models.embeddings import TokenEmbedding, PositionalEmbedding
from models.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        block_size=128,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.block_size = block_size

        # Convert token IDs into vectors
        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # Add information about token positions
        self.position_embedding = PositionalEmbedding(
            block_size,
            embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Stack multiple Transformer Blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize the final representations
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Convert representations into vocabulary scores
        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                "Input sequence exceeds the model's block_size"
            )

        # Embedding stage
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(token_ids)

        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)

        # Transformer processing
        x = self.transformer_blocks(x)

        # Final normalization and prediction
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        # Calculate loss when target tokens are provided
        loss = None

        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    "targets must have the same shape as token_ids"
                )

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

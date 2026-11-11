
import torch.nn as nn

from models.attention import CausalSelfAttention
from models.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout
        )

        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x):
        # First stage: attention with residual connection
        x = x + self.attention(self.layer_norm1(x))

        # Second stage: feed-forward with residual connection
        x = x + self.feed_forward(self.layer_norm2(x))

        return x

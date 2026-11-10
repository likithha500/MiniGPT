import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)
import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):

    def __init__(self, token_ids, block_size):
        self.data = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        self.block_size = block_size

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if len(self.data) <= block_size:
            raise ValueError(
                "Token sequence must be longer than block_size"
            )

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):

        x = self.data[index:index + self.block_size]

        y = self.data[
            index + 1:index + self.block_size + 1
        ]

        return x, y
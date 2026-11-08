
from torch.utils.data import DataLoader
from training.dataset import NextTokenDataset


def create_dataloaders(train_data, val_data, block_size, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_dataset = NextTokenDataset(train_data, block_size)
    val_dataset = NextTokenDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

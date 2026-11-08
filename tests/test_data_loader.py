
import pytest
import torch

from training.data_loader import create_dataloaders


@pytest.fixture
def sample_data():
    train_data = torch.arange(100)
    val_data = torch.arange(100, 150)
    return train_data, val_data


def test_dataloader_batch_shapes(sample_data):
    train_data, val_data = sample_data

    train_loader, val_loader = create_dataloaders(
        train_data, val_data, block_size=8, batch_size=4
    )

    x, y = next(iter(train_loader))

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_target_is_shifted_input(sample_data):
    train_data, val_data = sample_data

    train_loader, _ = create_dataloaders(
        train_data, val_data, block_size=8, batch_size=4
    )

    x, y = next(iter(train_loader))

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_validation_loader_preserves_order(sample_data):
    train_data, val_data = sample_data

    _, val_loader = create_dataloaders(
        train_data, val_data, block_size=8, batch_size=4
    )

    x, y = next(iter(val_loader))

    assert torch.equal(x[0], val_data[:8])
    assert torch.equal(y[0], val_data[1:9])


def test_invalid_batch_size(sample_data):
    train_data, val_data = sample_data

    with pytest.raises(ValueError):
        create_dataloaders(
            train_data, val_data, block_size=8, batch_size=0
        )


def test_dataloader_returns_tensors(sample_data):
    train_data, val_data = sample_data

    train_loader, _ = create_dataloaders(
        train_data, val_data, block_size=8, batch_size=4
    )

    x, y = next(iter(train_loader))

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

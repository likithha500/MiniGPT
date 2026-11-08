import pytest
import torch

from training.dataset import NextTokenDataset


def test_dataset_length():
    dataset = NextTokenDataset(
        [0, 1, 2, 3, 4],
        block_size=3
    )

    assert len(dataset) == 2


def test_input_sequence():
    dataset = NextTokenDataset(
        [0, 1, 2, 3, 4],
        block_size=3
    )

    x, y = dataset[0]

    assert x.tolist() == [0, 1, 2]


def test_target_sequence():
    dataset = NextTokenDataset(
        [0, 1, 2, 3, 4],
        block_size=3
    )

    x, y = dataset[0]

    assert y.tolist() == [1, 2, 3]


def test_input_and_target_are_tensors():
    dataset = NextTokenDataset(
        [0, 1, 2, 3, 4],
        block_size=3
    )

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_invalid_block_size():
    with pytest.raises(ValueError):
        NextTokenDataset([0, 1, 2], block_size=0)


def test_sequence_too_short():
    with pytest.raises(ValueError):
        NextTokenDataset([0, 1, 2], block_size=3)
import pytest
import torch

from models.feed_forward import FeedForward


def test_output_shape():
    model = FeedForward(embedding_dim=64)
    x = torch.randn(4, 8, 64)

    output = model(x)

    assert output.shape == (4, 8, 64)


def test_different_sequence_lengths():
    model = FeedForward(embedding_dim=64)
    x = torch.randn(2, 5, 64)

    output = model(x)

    assert output.shape == (2, 5, 64)


def test_backpropagation():
    model = FeedForward(embedding_dim=64)
    x = torch.randn(2, 8, 64, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_invalid_embedding_dimension():
    with pytest.raises(ValueError):
        FeedForward(embedding_dim=0)

import pytest

from training.config import GPTConfig


def test_default_configuration():
    config = GPTConfig()

    assert config.vocab_size == 65
    assert config.embedding_dim == 64
    assert config.num_heads == 4
    assert config.num_layers == 4
    assert config.batch_size == 32


def test_custom_configuration():
    config = GPTConfig(
        vocab_size=100,
        embedding_dim=128,
        num_heads=8,
        num_layers=6
    )

    assert config.vocab_size == 100
    assert config.embedding_dim == 128
    assert config.num_heads == 8
    assert config.num_layers == 6


def test_invalid_embedding_dimension():
    with pytest.raises(ValueError):
        GPTConfig(embedding_dim=63, num_heads=4)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        GPTConfig(batch_size=0)


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        GPTConfig(learning_rate=0)


def test_invalid_dropout():
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0)


def test_invalid_number_of_layers():
    with pytest.raises(ValueError):
        GPTConfig(num_layers=0)

import pytest
import torch

from data.preprocess import prepare_data


def test_vocabulary_size():
    text = "hello world"

    tokenizer, train_data, val_data = prepare_data(text)

    assert tokenizer.vocab_size == 8


def test_total_token_count():
    text = "hello world"

    tokenizer, train_data, val_data = prepare_data(text)

    assert len(train_data) + len(val_data) == len(text)


def test_train_validation_split():
    text = "abcdefghijklmnopqrst"

    tokenizer, train_data, val_data = prepare_data(text)

    assert len(train_data) == 18
    assert len(val_data) == 2


def test_output_types():
    tokenizer, train_data, val_data = prepare_data("hello world")

    assert isinstance(train_data, torch.Tensor)
    assert isinstance(val_data, torch.Tensor)


def test_invalid_train_ratio():
    with pytest.raises(ValueError):
        prepare_data("hello world", train_ratio=1.0)


def test_data_order_preserved():
    text = "abcdefghijklmnopqrst"

    tokenizer, train_data, val_data = prepare_data(text)

    reconstructed = tokenizer.decode(
        train_data.tolist() + val_data.tolist()
    )

    assert reconstructed == text

import pytest
import torch

from models.gpt import GPTLanguageModel


@pytest.fixture
def model():
    return GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=128,
        num_layers=2,
        dropout=0.0
    )


def test_output_shape(model):
    token_ids = torch.randint(0, 65, (4, 8))

    logits, loss = model(token_ids)

    assert logits.shape == (4, 8, 65)
    assert loss is None


def test_loss_calculation(model):
    token_ids = torch.randint(0, 65, (4, 8))
    targets = torch.randint(0, 65, (4, 8))

    logits, loss = model(token_ids, targets)

    assert logits.shape == (4, 8, 65)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_backpropagation(model):
    token_ids = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 8))

    logits, loss = model(token_ids, targets)
    loss.backward()

    assert model.token_embedding.embedding.weight.grad is not None
    assert model.language_model_head.weight.grad is not None


def test_sequence_exceeds_block_size():
    model = GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=8,
        num_layers=2
    )

    token_ids = torch.randint(0, 65, (2, 9))

    with pytest.raises(ValueError):
        model(token_ids)


def test_invalid_target_shape(model):
    token_ids = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 7))

    with pytest.raises(ValueError):
        model(token_ids, targets)


def test_invalid_number_of_layers():
    with pytest.raises(ValueError):
        GPTLanguageModel(
            vocab_size=65,
            num_layers=0
        )

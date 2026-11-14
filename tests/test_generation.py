
import pytest
import torch

from models.gpt import GPTLanguageModel


@pytest.fixture
def model():
    return GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=16,
        num_layers=2,
        dropout=0.0
    )


def test_generation_output_shape(model):
    prompt = torch.randint(0, 65, (1, 5))

    output = model.generate(
        prompt,
        max_new_tokens=10
    )

    assert output.shape == (1, 15)


def test_top_k_generation(model):
    prompt = torch.randint(0, 65, (1, 5))

    output = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=5
    )

    assert output.shape == (1, 15)


def test_invalid_temperature(model):
    prompt = torch.randint(0, 65, (1, 5))

    with pytest.raises(ValueError):
        model.generate(prompt, 10, temperature=0)


def test_invalid_top_k(model):
    prompt = torch.randint(0, 65, (1, 5))

    with pytest.raises(ValueError):
        model.generate(prompt, 10, top_k=0)


def test_top_k_exceeds_vocabulary(model):
    prompt = torch.randint(0, 65, (1, 5))

    with pytest.raises(ValueError):
        model.generate(prompt, 10, top_k=66)


def test_generation_preserves_prompt(model):
    prompt = torch.randint(0, 65, (1, 5))

    output = model.generate(
        prompt,
        max_new_tokens=10,
        top_k=5
    )

    assert torch.equal(output[:, :5], prompt)

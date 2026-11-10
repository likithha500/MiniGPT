
import pytest
import torch

from models.attention import CausalSelfAttention


def test_attention_output_shape():
    model = CausalSelfAttention(64, 4, 128)

    x = torch.randn(4, 8, 64)
    output = model(x)

    assert output.shape == (4, 8, 64)


def test_attention_supports_shorter_sequences():
    model = CausalSelfAttention(64, 4, 128)

    x = torch.randn(2, 5, 64)
    output = model(x)

    assert output.shape == (2, 5, 64)


def test_attention_rejects_invalid_head_dimensions():
    with pytest.raises(ValueError):
        CausalSelfAttention(63, 4, 128)


def test_attention_rejects_sequence_longer_than_block_size():
    model = CausalSelfAttention(64, 4, 8)

    x = torch.randn(1, 9, 64)

    with pytest.raises(ValueError):
        model(x)


def test_attention_blocks_future_information():
    model = CausalSelfAttention(
        embedding_dim=64,
        num_heads=4,
        block_size=8,
        dropout=0.0
    )

    model.eval()

    x1 = torch.randn(1, 5, 64)
    x2 = x1.clone()

    # Modify only the final token
    x2[:, 4, :] += 100

    with torch.no_grad():
        output1 = model(x1)
        output2 = model(x2)

    # Earlier positions must not change
    assert torch.allclose(
        output1[:, :4, :],
        output2[:, :4, :],
        atol=1e-6
    )


def test_attention_supports_backpropagation():
    model = CausalSelfAttention(64, 4, 128)

    x = torch.randn(2, 8, 64, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert model.query.weight.grad is not None
    assert model.key.weight.grad is not None
    assert model.value.weight.grad is not None

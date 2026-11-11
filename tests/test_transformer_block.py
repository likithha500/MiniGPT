
import torch

from models.transformer_block import TransformerBlock


def test_output_shape():
    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        block_size=128
    )

    x = torch.randn(4, 8, 64)
    output = model(x)

    assert output.shape == (4, 8, 64)


def test_different_sequence_lengths():
    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        block_size=128
    )

    x = torch.randn(2, 16, 64)
    output = model(x)

    assert output.shape == (2, 16, 64)


def test_residual_connection_preserves_shape():
    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        block_size=128
    )

    x = torch.randn(2, 8, 64)
    output = model(x)

    assert output.shape == x.shape


def test_backpropagation():
    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        block_size=128
    )

    x = torch.randn(2, 8, 64, requires_grad=True)
    output = model(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_causal_behavior():
    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        block_size=128,
        dropout=0.0
    )

    model.eval()

    x1 = torch.randn(1, 8, 64)
    x2 = x1.clone()

    # Change only the final token
    x2[:, -1, :] = torch.randn(1, 64)

    with torch.no_grad():
        output1 = model(x1)
        output2 = model(x2)

    # Earlier positions must not be affected
    assert torch.allclose(
        output1[:, :-1, :],
        output2[:, :-1, :],
        atol=1e-6
    )


import torch

from models.embeddings import TokenEmbedding, PositionalEmbedding


def test_token_embedding_shape():
    layer = TokenEmbedding(vocab_size=65, embedding_dim=64)

    token_ids = torch.randint(0, 65, (4, 8))
    output = layer(token_ids)

    assert output.shape == (4, 8, 64)


def test_positional_embedding_shape():
    layer = PositionalEmbedding(block_size=128, embedding_dim=64)

    token_ids = torch.randint(0, 65, (4, 8))
    output = layer(token_ids)

    assert output.shape == (8, 64)


def test_same_token_has_same_embedding():
    layer = TokenEmbedding(vocab_size=65, embedding_dim=64)

    token_ids = torch.tensor([[5, 5]])
    output = layer(token_ids)

    assert torch.equal(output[:, 0, :], output[:, 1, :])


def test_positional_embeddings_are_different():
    layer = PositionalEmbedding(block_size=128, embedding_dim=64)

    token_ids = torch.randint(0, 65, (4, 8))
    output = layer(token_ids)

    assert not torch.equal(output[0], output[1])


def test_combined_embedding_shape():
    token_layer = TokenEmbedding(65, 64)
    position_layer = PositionalEmbedding(128, 64)

    token_ids = torch.randint(0, 65, (4, 8))

    token_vectors = token_layer(token_ids)
    position_vectors = position_layer(token_ids)

    combined = token_vectors + position_vectors

    assert combined.shape == (4, 8, 64)

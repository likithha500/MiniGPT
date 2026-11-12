
import pytest
import torch

from models.gpt import GPTLanguageModel
from training.checkpoint import save_checkpoint, load_checkpoint


def create_model_and_optimizer():
    model = GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=16,
        num_layers=2,
        dropout=0.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    return model, optimizer


def test_checkpoint_file_is_created(tmp_path):
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = tmp_path / "checkpoints" / "model.pt"

    save_checkpoint(
        model,
        optimizer,
        iteration=100,
        history={"train_loss": [2.5]},
        file_path=checkpoint_path
    )

    assert checkpoint_path.exists()


def test_checkpoint_restores_model_weights(tmp_path):
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = tmp_path / "model.pt"

    original_weights = (
        model.token_embedding.embedding.weight.detach().clone()
    )

    save_checkpoint(
        model, optimizer, 100, {}, checkpoint_path
    )

    with torch.no_grad():
        model.token_embedding.embedding.weight.add_(1.0)

    load_checkpoint(model, optimizer, checkpoint_path)

    restored_weights = (
        model.token_embedding.embedding.weight.detach()
    )

    assert torch.equal(original_weights, restored_weights)


def test_checkpoint_restores_metadata(tmp_path):
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = tmp_path / "model.pt"

    history = {
        "iterations": [100],
        "train_loss": [2.5],
        "val_loss": [2.8]
    }

    config = {"learning_rate": 0.001}

    save_checkpoint(
        model,
        optimizer,
        iteration=100,
        history=history,
        file_path=checkpoint_path,
        config=config
    )

    metadata = load_checkpoint(
        model, optimizer, checkpoint_path
    )

    assert metadata["iteration"] == 100
    assert metadata["history"] == history
    assert metadata["config"] == config


def test_missing_checkpoint_raises_error(tmp_path):
    model, optimizer = create_model_and_optimizer()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            model,
            optimizer,
            tmp_path / "missing.pt"
        )


def test_invalid_iteration_raises_error(tmp_path):
    model, optimizer = create_model_and_optimizer()

    with pytest.raises(ValueError):
        save_checkpoint(
            model,
            optimizer,
            iteration=-1,
            history={},
            file_path=tmp_path / "model.pt"
        )

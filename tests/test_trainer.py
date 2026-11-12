import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.gpt import GPTLanguageModel
from training.trainer import Trainer


def test_train_step_returns_loss():
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

    trainer = Trainer(
        model=model,
        train_loader=None,
        optimizer=optimizer
    )

    inputs = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 8))

    loss = trainer.train_step(inputs, targets)

    assert isinstance(loss, float)
    assert loss > 0


def test_train_step_updates_parameters():
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

    trainer = Trainer(model, None, optimizer)

    inputs = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 8))

    original_weights = (
        model.token_embedding.embedding.weight.detach().clone()
    )

    trainer.train_step(inputs, targets)

    updated_weights = model.token_embedding.embedding.weight.detach()

    assert not torch.equal(original_weights, updated_weights)


def test_model_enters_training_mode():
    model = GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=16,
        num_layers=2
    )

    optimizer = torch.optim.AdamW(model.parameters())

    trainer = Trainer(model, None, optimizer)

    inputs = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 8))

    trainer.train_step(inputs, targets)

    assert model.training





def create_trainer_with_data():
    model = GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=16,
        num_layers=2,
        dropout=0.0
    )

    inputs = torch.randint(0, 65, (4, 8))
    targets = torch.randint(0, 65, (4, 8))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    return Trainer(model, loader, optimizer)


def test_full_training_loop():
    trainer = create_trainer_with_data()

    losses = trainer.train(
        max_iterations=5,
        log_interval=2
    )

    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)


def test_training_restarts_dataloader():
    trainer = create_trainer_with_data()

    # Only two batches are available, but we request five iterations.
    losses = trainer.train(max_iterations=5, log_interval=5)

    assert len(losses) == 5


def test_training_rejects_invalid_iterations():
    trainer = create_trainer_with_data()

    with pytest.raises(ValueError):
        trainer.train(max_iterations=0)


def test_training_requires_dataloader():
    model = GPTLanguageModel(
        vocab_size=65,
        embedding_dim=64,
        num_heads=4,
        block_size=16,
        num_layers=2
    )

    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, None, optimizer)

    with pytest.raises(ValueError):
        trainer.train(max_iterations=5)


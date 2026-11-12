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


def create_validation_loader():
    inputs = torch.randint(0, 65, (6, 8))
    targets = torch.randint(0, 65, (6, 8))

    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)


def test_validation_returns_loss():
    trainer = create_trainer_with_data()

    val_loader = create_validation_loader()
    val_loss = trainer.evaluate(val_loader)

    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_validation_does_not_update_weights():
    trainer = create_trainer_with_data()
    val_loader = create_validation_loader()

    original_weights = (
        trainer.model.token_embedding.embedding.weight.detach().clone()
    )

    trainer.evaluate(val_loader)

    updated_weights = (
        trainer.model.token_embedding.embedding.weight.detach()
    )

    assert torch.equal(original_weights, updated_weights)


def test_validation_restores_training_mode():
    trainer = create_trainer_with_data()
    val_loader = create_validation_loader()

    trainer.model.train()
    trainer.evaluate(val_loader)

    assert trainer.model.training


def test_validation_rejects_invalid_batches():
    trainer = create_trainer_with_data()
    val_loader = create_validation_loader()

    with pytest.raises(ValueError):
        trainer.evaluate(val_loader, eval_batches=0)

def test_training_records_loss_history():
    trainer = create_trainer_with_data()
    val_loader = create_validation_loader()

    trainer.train(
        max_iterations=4,
        log_interval=2,
        val_loader=val_loader,
        eval_interval=2,
        eval_batches=2
    )

    assert len(trainer.history["iterations"]) == 2
    assert len(trainer.history["train_loss"]) == 2
    assert len(trainer.history["val_loss"]) == 2


def test_training_history_records_correct_iterations():
    trainer = create_trainer_with_data()
    val_loader = create_validation_loader()

    trainer.train(
        max_iterations=6,
        log_interval=2,
        val_loader=val_loader,
        eval_interval=3,
        eval_batches=2
    )

    assert trainer.history["iterations"] == [3, 6]


def test_training_without_validation():
    trainer = create_trainer_with_data()

    trainer.train(max_iterations=3, log_interval=2)

    assert trainer.history["iterations"] == []
    assert trainer.history["val_loss"] == []

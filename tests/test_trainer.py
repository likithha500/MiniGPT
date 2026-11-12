
import torch

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


from dataclasses import asdict
from pathlib import Path

import torch

from data.preprocess import load_text, prepare_data
from models.gpt import GPTLanguageModel
from training.config import GPTConfig
from training.data_loader import create_dataloaders
from training.trainer import Trainer
from training.checkpoint import save_checkpoint


def run_training(max_iterations=100):
    # Find the project root directory
    project_root = Path(__file__).resolve().parents[1]

    # Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # Load and preprocess Tiny Shakespeare
    dataset_path = project_root / "data" / "raw" / "tinyshakespeare.txt"
    text = load_text(dataset_path)

    tokenizer, train_data, val_data = prepare_data(text)

    # Create configuration
    config = GPTConfig(vocab_size=tokenizer.vocab_size)

    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Training tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        block_size=config.block_size,
        batch_size=config.batch_size
    )

    # Initialize GPT
    model = GPTLanguageModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        block_size=config.block_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device
    )

    # Train and validate
    training_losses = trainer.train(
        max_iterations=max_iterations,
        log_interval=10,
        val_loader=val_loader,
        eval_interval=50,
        eval_batches=config.eval_batches
    )

    # Save checkpoint
    checkpoint_path = project_root / "checkpoints" / "minigpt_checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=max_iterations,
        history=trainer.history,
        file_path=checkpoint_path,
        config=asdict(config)
    )

    print(f"\nTraining completed!")
    print(f"Final training loss: {training_losses[-1]:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return model, trainer, tokenizer


if __name__ == "__main__":
    run_training(max_iterations=100)

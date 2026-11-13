
from dataclasses import asdict
from pathlib import Path

import torch

from data.preprocess import load_text, prepare_data
from models.gpt import GPTLanguageModel
from training.config import GPTConfig
from training.data_loader import create_dataloaders
from training.trainer import Trainer
from training.checkpoint import save_checkpoint, load_checkpoint


def run_training(
    max_iterations=5000,
    checkpoint_dir=None,
    resume=True
):
    project_root = Path(__file__).resolve().parents[1]

    if checkpoint_dir is None:
        checkpoint_dir = project_root / "checkpoints"

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "minigpt_checkpoint.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    text = load_text(
        project_root / "data" / "raw" / "tinyshakespeare.txt"
    )

    tokenizer, train_data, val_data = prepare_data(text)

    config = GPTConfig(vocab_size=tokenizer.vocab_size)

    train_loader, val_loader = create_dataloaders(
        train_data,
        val_data,
        config.block_size,
        config.batch_size
    )

    model = GPTLanguageModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        block_size=config.block_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    start_iteration = 0
    history = None

    if resume and checkpoint_path.exists():
        metadata = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            device=device
        )

        start_iteration = metadata["iteration"]
        history = metadata["history"]

        if metadata["config"] != asdict(config):
            raise ValueError(
                "Saved checkpoint configuration does not match current configuration"
            )

        print(f"Resuming from iteration {start_iteration}")

    trainer = Trainer(
        model,
        train_loader,
        optimizer,
        device=device
    )

    def checkpoint_callback(iteration, current_history):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            history=current_history,
            file_path=checkpoint_path,
            config=asdict(config)
        )

        print(f"Checkpoint saved at iteration {iteration}")

    if start_iteration >= max_iterations:
        print("The checkpoint has already reached the target iteration.")
        return model, trainer, tokenizer

    trainer.train(
        max_iterations=max_iterations,
        log_interval=100,
        val_loader=val_loader,
        eval_interval=500,
        eval_batches=config.eval_batches,
        start_iteration=start_iteration,
        history=history,
        checkpoint_interval=500,
        checkpoint_callback=checkpoint_callback
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=max_iterations,
        history=trainer.history,
        file_path=checkpoint_path,
        config=asdict(config)
    )

    print(f"Training completed. Final checkpoint: {checkpoint_path}")

    return model, trainer, tokenizer


if __name__ == "__main__":
    run_training(max_iterations=5000)

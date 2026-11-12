
from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    iteration,
    history,
    file_path,
    config=None
):
    if iteration < 0:
        raise ValueError("iteration cannot be negative")

    checkpoint_path = Path(file_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "history": history,
        "config": config
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model,
    optimizer,
    file_path,
    device="cpu"
):
    checkpoint_path = Path(file_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "iteration",
        "history",
        "config"
    }

    if not required_keys.issubset(checkpoint):
        raise ValueError("Invalid checkpoint format")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "iteration": checkpoint["iteration"],
        "history": checkpoint["history"],
        "config": checkpoint["config"]
    }

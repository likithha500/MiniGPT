from pathlib import Path
import torch

from tokenizer.tokenizer import CharacterTokenizer


def load_text(file_path):
    """Read the raw Shakespeare text."""

    return Path(file_path).read_text(encoding="utf-8")


def prepare_data(text, train_ratio=0.9):
    """Tokenize text and split into training and validation data."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Build tokenizer using the complete corpus vocabulary
    tokenizer = CharacterTokenizer(text)

    # Convert complete text into token IDs
    token_ids = tokenizer.encode(text)

    # Convert token IDs into a PyTorch tensor
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Calculate the split position
    split_index = int(len(token_ids) * train_ratio)

    # Split without shuffling to preserve text order
    train_data = token_ids[:split_index]
    val_data = token_ids[split_index:]

    return tokenizer, train_data, val_data


if __name__ == "__main__":

    file_path = "data/raw/tinyshakespeare.txt"

    text = load_text(file_path)

    tokenizer, train_data, val_data = prepare_data(text)

    print("Dataset preprocessing complete!")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Total tokens:", len(train_data) + len(val_data))
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
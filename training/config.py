
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Model configuration
    vocab_size: int = 65
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    block_size: int = 128
    dropout: float = 0.1

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iterations: int = 5000
    eval_interval: int = 500
    eval_batches: int = 20

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

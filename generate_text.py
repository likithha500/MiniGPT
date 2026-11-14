
from pathlib import Path
import torch

from data.preprocess import load_text, prepare_data
from models.gpt import GPTLanguageModel
from training.checkpoint import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load original dataset to reconstruct the tokenizer
text = load_text(
    PROJECT_ROOT / "data" / "raw" / "tinyshakespeare.txt"
)

tokenizer, _, _ = prepare_data(text)

# Recreate the model with matching architecture
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    num_heads=4,
    block_size=128,
    num_layers=4,
    dropout=0.1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003
)

# Restore trained weights
checkpoint_path = (
    PROJECT_ROOT / "checkpoints" / "minigpt_checkpoint.pt"
)

metadata = load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device=device
)

model.to(device)
model.eval()

# Give the model a starting prompt
prompt = "ROMEO:"

prompt_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

# Generate 500 new characters
generated_ids = model.generate(
    prompt_ids,
    max_new_tokens=500
)

generated_text = tokenizer.decode(
    generated_ids[0].tolist()
)

print("\nGenerated Shakespeare-style text:\n")
print(generated_text)

# Generate with balanced sampling
generated_1 = model.generate(
    prompt_ids.clone(),
    max_new_tokens=500,
    temperature=0.8,
    top_k=10
)

# Generate with more predictable sampling
generated_2 = model.generate(
    prompt_ids.clone(),
    max_new_tokens=500,
    temperature=0.6,
    top_k=5
)

# Generate with more diversity
generated_3 = model.generate(
    prompt_ids.clone(),
    max_new_tokens=500,
    temperature=1.0,
    top_k=20
)

print("\n--- Balanced ---\n")
print(tokenizer.decode(generated_1[0].tolist()))

print("\n--- Predictable ---\n")
print(tokenizer.decode(generated_2[0].tolist()))

print("\n--- Diverse ---\n")
print(tokenizer.decode(generated_3[0].tolist()))

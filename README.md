# MiniGPT — From Scratch

A decoder-only Transformer language model implemented from scratch using Python and PyTorch.

The objective of this project is to understand the internal architecture of GPT-style language models by building, training, evaluating and generating text with a small language model.

## Project Objectives

- Understand tokenization and vocabulary creation.
- Implement next-token prediction.
- Build token embeddings and positional embeddings.
- Implement causal self-attention.
- Develop Transformer blocks.
- Train a decoder-only language model.
- Evaluate model performance.
- Generate text using the trained model.

## Tech Stack

- **Language:** Python 3.13
- **Deep Learning:** PyTorch
- **Numerical Computing:** NumPy
- **Testing:** Pytest
- **Visualization:** Matplotlib

## Dataset

This project uses the Tiny Shakespeare dataset for character-level language modeling experiments.

- Total characters: 1,115,394
- Vocabulary size: 65 unique characters
- Training split: 90%
- Validation split: 10%

## Project Architecture

```text
MiniGPT/
├── data/
│   ├── raw/
│   ├── processed/
│   └── preprocess.py
├── tokenizer/
│   └── tokenizer.py
├── training/
│   ├── dataset.py
│   └── data_loader.py
├── tests/
├── configs/
├── model/
├── evaluation/
└── generation/
```

## Getting Started

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MiniGPT
```

Create and activate a virtual environment:

```bash
py -3.13 -m venv .venv
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Development Progress

- [x] Repository and environment setup
- [x] Character-level tokenizer
- [x] Next-token dataset
- [x] Tiny Shakespeare preprocessing
- [x] Training and validation split
- [x] Batching and DataLoader
- [x] Token embeddings
- [x] Positional embeddings
- [x] Causal self-attention
- [x] Transformer blocks
- [x] GPT model
- [x] Training and evaluation
- [ ] Text generation
- [ ] Long-duration training
- [ ] Training resumption

## Learning Goal

This project is an educational implementation designed to understand how GPT-style language models work internally. It is not intended to reproduce the scale or capabilities of production LLMs

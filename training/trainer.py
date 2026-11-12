

import torch


class Trainer:
    def __init__(self, model, train_loader, optimizer, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = torch.device(device)

        self.model.to(self.device)

    def train_step(self, inputs, targets):
        self.model.train()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Clear previous gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits, loss = self.model(inputs, targets)

        # Backward pass
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        return loss.item()

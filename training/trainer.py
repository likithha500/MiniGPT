
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

        self.optimizer.zero_grad(set_to_none=True)

        logits, loss = self.model(inputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, max_iterations, log_interval=100):
        if self.train_loader is None:
            raise ValueError("train_loader must be provided")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if log_interval <= 0:
            raise ValueError("log_interval must be positive")

        self.model.train()

        data_iterator = iter(self.train_loader)
        training_losses = []

        for iteration in range(1, max_iterations + 1):

            try:
                inputs, targets = next(data_iterator)

            except StopIteration:
                data_iterator = iter(self.train_loader)
                inputs, targets = next(data_iterator)

            loss = self.train_step(inputs, targets)
            training_losses.append(loss)

            if iteration % log_interval == 0:
                average_loss = sum(
                    training_losses[-log_interval:]
                ) / log_interval

                print(
                    f"Iteration {iteration}/{max_iterations} | "
                    f"Average Loss: {average_loss:.4f}"
                )

        return training_losses

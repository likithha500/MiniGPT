
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

    
    def train(
        self,
        max_iterations,
        log_interval=100,
        val_loader=None,
        eval_interval=100,
        eval_batches=20
    ):
        if self.train_loader is None:
            raise ValueError("train_loader must be provided")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if log_interval <= 0:
            raise ValueError("log_interval must be positive")

        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if eval_batches <= 0:
            raise ValueError("eval_batches must be positive")

        self.model.train()

        data_iterator = iter(self.train_loader)
        training_losses = []

        # Store training and validation history
        self.history = {
            "iterations": [],
            "train_loss": [],
            "val_loss": []
        }

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
                    f"Training Loss: {average_loss:.4f}"
                )

            if val_loader is not None and iteration % eval_interval == 0:
                val_loss = self.evaluate(
                    val_loader,
                    eval_batches=eval_batches
                )

                self.history["iterations"].append(iteration)
                self.history["train_loss"].append(average_loss if iteration % log_interval == 0 else sum(training_losses[-min(iteration, log_interval):]) / min(iteration, log_interval))
                self.history["val_loss"].append(val_loss)

                print(f"Validation Loss: {val_loss:.4f}")

        return training_losses

    
    def evaluate(self, val_loader, eval_batches=20):
        if val_loader is None:
            raise ValueError("val_loader must be provided")

        if eval_batches <= 0:
            raise ValueError("eval_batches must be positive")

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        try:
            with torch.no_grad():
                for batch_index, (inputs, targets) in enumerate(val_loader):
                    if batch_index >= eval_batches:
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    logits, loss = self.model(inputs, targets)

                    total_loss += loss.item()
                    num_batches += 1

            if num_batches == 0:
                raise ValueError("val_loader contains no batches")

            return total_loss / num_batches

        finally:
            self.model.train(was_training)


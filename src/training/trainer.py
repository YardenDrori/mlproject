"""Training loop and utilities."""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement required
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


class Trainer:
    """Training loop for image classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 5
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Torch device
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping (0 to disable)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.best_val_acc = 0.0

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Early stopping
        self.early_stopping = None
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(patience=early_stopping_patience)

    def train_epoch(self) -> Tuple[float, float]:
        """
        Run a single training epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Run validation.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, epochs: int) -> dict:
        """
        Full training loop.

        Args:
            epochs: Number of epochs to train

        Returns:
            Training history dict
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pth")
                print(f"New best model saved! (Val Acc: {val_acc:.4f})")

            # Early stopping
            if self.early_stopping and self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Save final model
        self.save_checkpoint("final_model.pth")
        print(f"\nTraining complete! Best Val Acc: {self.best_val_acc:.4f}")

        return history

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

#!/usr/bin/env python3
"""
Train the freshness classifier model.

Usage:
    python train_freshness_classifier.py --data_dir data/freshness --epochs 30

Expected data structure:
    data/freshness/
        train/
            fresh/
            rotten/
        val/
            fresh/
            rotten/
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.freshness_classifier import FreshnessClassifier
from src.data.dataloader import create_dataloader
from src.training.trainer import Trainer
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train freshness classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/freshness",
        help="Path to data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone weights"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
        help="Early stopping patience (0 to disable)"
    )
    args = parser.parse_args()

    # Get device
    device = get_device()

    # Check data directory exists
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        print("\nExpected structure:")
        print("  data/freshness/")
        print("      train/")
        print("          fresh/")
        print("          rotten/")
        print("      val/")
        print("          fresh/")
        print("          rotten/")
        return

    # Create dataloaders
    print("\nLoading data...")
    train_loader, classes = create_dataloader(
        train_dir,
        batch_size=args.batch_size,
        is_train=True,
        img_size=args.img_size,
        device=device
    )
    val_loader, _ = create_dataloader(
        val_dir,
        batch_size=args.batch_size,
        is_train=False,
        img_size=args.img_size,
        device=device
    )

    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = FreshnessClassifier(
        num_classes=len(classes),
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    print(f"Trainable parameters: {model.get_num_params():,}")

    # Training setup
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Create checkpoint directory
    checkpoint_dir = "checkpoints/freshness_classifier"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train
    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.early_stopping
    )

    history = trainer.fit(args.epochs)

    # Save class mapping
    torch.save({"classes": classes}, os.path.join(checkpoint_dir, "classes.pth"))
    print(f"\nClass mapping saved to {checkpoint_dir}/classes.pth")


if __name__ == "__main__":
    main()

"""DataLoader factory for creating train/val dataloaders."""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms
from ..utils.device import get_num_workers


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    is_train: bool = True,
    num_workers: int = None,
    img_size: int = 224,
    device: torch.device = None
) -> Tuple[DataLoader, List[str]]:
    """
    Create a DataLoader from an image folder.

    Expects folder structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...

    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for training
        is_train: If True, apply training augmentations
        num_workers: Number of worker processes (auto-detected if None)
        img_size: Target image size
        device: Torch device (used to determine num_workers if not specified)

    Returns:
        Tuple of (DataLoader, list of class names)
    """
    transform = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)

    dataset = ImageFolder(root=data_dir, transform=transform)

    if num_workers is None:
        if device is None:
            device = torch.device("cpu")
        num_workers = get_num_workers(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    return loader, dataset.classes

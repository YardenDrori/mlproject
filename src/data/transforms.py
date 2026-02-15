"""Data augmentation transforms for training and validation."""

from torchvision import transforms


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get training data transforms with augmentation.

    Includes random cropping, flipping, rotation, and color jitter
    to improve model generalization.

    Args:
        img_size: Target image size (default 224 for EfficientNet)

    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get validation/inference transforms (no augmentation).

    Args:
        img_size: Target image size (default 224 for EfficientNet)

    Returns:
        Composed transforms for validation/inference
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

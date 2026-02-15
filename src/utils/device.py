"""Device detection utility for CUDA, MPS, and CPU."""

import torch


def get_device() -> torch.device:
    """
    Detect the best available device for training/inference.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_num_workers(device: torch.device) -> int:
    """
    Get optimal number of workers for DataLoader based on device.

    MPS can have issues with multiprocessing, so we use 0 workers.

    Args:
        device: The torch device being used

    Returns:
        int: Number of workers to use
    """
    import os
    if device.type == "mps":
        return 0
    return min(4, os.cpu_count() or 1)

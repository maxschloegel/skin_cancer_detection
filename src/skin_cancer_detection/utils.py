import torch
from torch.utils.data import random_split


def train_val_split(dataset, val_split=0.2, seed=None):
    """
    Splits a PyTorch dataset into training and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        val_split (float): The fraction of the dataset to include in the
            validation set (default: 0.2).
        seed (int): Seed for random number generation (default: None).

    Returns:
        train_dataset (torch.utils.data.Dataset): Dataset for the
            training set.
        val_dataset (torch.utils.data.Dataset): Dataset for the
            validation set.
    """
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate sizes for the split
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    # Split the dataset into train and validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

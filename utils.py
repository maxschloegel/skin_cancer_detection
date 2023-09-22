import torch
from torch.utils.data import random_split


def train_val_split(dataset, val_split=0.2, seed=None):
    """
    Splits a PyTorch dataset into training and validation sets.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        val_split (float): The fraction of the dataset to include in the validation set (default: 0.2).
        batch_size (int): Batch size for data loaders (default: 32).
        num_workers (int): Number of CPU processes to use for data loading (default: 4).
        shuffle (bool): Whether to shuffle the data before splitting (default: True).
        seed (int): Seed for random number generation (default: None).

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
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


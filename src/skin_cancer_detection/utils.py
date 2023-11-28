import os
from dotenv import load_dotenv
from hydra import compose, initialize
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import transforms


load_dotenv()


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


def load_model(model_name="scd_model_prod", stage="Production"):
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pytorch.load_model(model_uri)
    return model


def get_label_for_prediction(predictions, class_strings):
    prediction_softmax = F.softmax(predictions, dim=1)
    idx = predictions.argmax().item()
    return class_strings[idx], prediction_softmax[0][idx]


def apply_transforms(transform, image):
    """Applys ToTensor and the given transformations

    Since we want to make the transformations scriptable, we need to take
    `ToTensor()` out of the nn.Sequential(transformations). To have a
    standardized way of applying transformations and not forgetting 
    `ToTensor()` we will use this function for transformations.
    """
    tensor = transforms.ToTensor()(image)
    tensor = transform(tensor)
    return tensor


#def load_config():
#    with initialize(version_base=None, config_path="../../conf"):
#        cfg = compose(config_name="training_config")
#    return cfg

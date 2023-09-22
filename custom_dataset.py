import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, csv_file, image_column="lesion_id", label_column="dx", transform=None):
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_file)
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

        # Create a dictionary to map labels to integers
        self.label_to_int = {label: idx for idx, label in enumerate(self.data_info[label_column].unique())}

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"{self.data_info.loc[idx, self.image_column]}.jpg")
        image = Image.open(img_name)
        label = self.label_to_int[self.data_info.loc[idx, self.label_column]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_info)

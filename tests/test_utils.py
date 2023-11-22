from torchvision.datasets import FakeData

from skin_cancer_detection.utils import train_val_split


def test_train_val_split():
    data = FakeData()

    train_data, val_data = train_val_split(data)

    assert len(train_data) == 800
    assert len(val_data) == 200

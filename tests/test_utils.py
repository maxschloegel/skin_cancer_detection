from torchvision.datasets import FakeData

from skin_cancer_detection.utils import train_val_split


def test_train_val_split(data):
    train, val = train_val_split(data)

    assert len(train) == 800
    assert len(val) == 200


def test_train_val_split_seed_difference(data):
    train_1, val_1 = train_val_split(data, seed=1)
    train_2, val_2 = train_val_split(data, seed=2)

    assert train_1.indices != train_2.indices
    assert val_1.indices != val_2.indices


def test_train_val_split_sum_complete(data):
    train, val = train_val_split(data)

    assert len(train) + len(val) == len(data)

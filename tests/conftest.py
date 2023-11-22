import pytest
from torchvision.datasets import FakeData


@pytest.fixture(name="data")
def create_fake_data():
    data = FakeData()
    return data

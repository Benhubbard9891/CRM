import pytest
import torch

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def embedding_dim():
    return 32

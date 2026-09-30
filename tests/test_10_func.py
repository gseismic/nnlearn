import config
import time
import numpy as np
import nnlearn
import torch
from nnlearn import functional as F
from nnlearn import Tensor, allclose
import nnlearn.nn as nn
from nnlearn.optim import SGD


def test_func_basic():
    A = nnlearn.linspace(0, 1, 11)
    A_torch = torch.linspace(0, 1, 11)
    print(repr(A.numpy()))
    print(repr(A_torch.numpy()))
    assert np.allclose(A.numpy(), A_torch.numpy())

    A = nnlearn.unsqueeze(nnlearn.linspace(0, 1, 11), dim=0)
    A_torch = torch.unsqueeze(torch.linspace(0, 1, 11), dim=0)
    print(f'{A=}')
    print(f'{A_torch=}')
    assert np.allclose(A.numpy(), A_torch.numpy())

    A = nnlearn.normal(0, 1, size=(3, 5))
    A_torch = torch.normal(0, 1, size=(3, 5))
    print(f'{A=}')
    print(f'{A_torch=}')
    assert A.numpy().shape == A_torch.numpy().shape


if __name__ == '__main__':
    if 1: 
        test_func_basic()

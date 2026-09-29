"""本书第三篇的小型神经网络层与训练组件。"""

import sys
from pathlib import Path


_tensor_code = Path(__file__).resolve().parents[2] / "02-Tensor与自动微分"
if str(_tensor_code) not in sys.path:
    sys.path.insert(0, str(_tensor_code))

from mini_tensor import Tensor

from .core import Module, Parameter
from .layers import Linear, ReLU, Sequential, Sigmoid, Tanh
from .losses import CrossEntropyLoss, MSELoss
from .optim import SGD
from .data import ArrayDataset, DataLoader
from .spatial import Conv2d, Flatten, MaxPool2d
from .attention import (
    Embedding,
    LayerNorm,
    MultiheadSelfAttention,
    TransformerEncoderLayer,
)
from .checkpoint import load_state_dict, save_state_dict

__all__ = [
    "ArrayDataset",
    "CrossEntropyLoss",
    "Conv2d",
    "DataLoader",
    "Embedding",
    "Flatten",
    "LayerNorm",
    "Linear",
    "MSELoss",
    "MaxPool2d",
    "Module",
    "MultiheadSelfAttention",
    "Parameter",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "Tensor",
    "TransformerEncoderLayer",
    "load_state_dict",
    "save_state_dict",
]

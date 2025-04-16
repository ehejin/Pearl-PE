from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/mlp.py 

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))  
        gate = torch.sigmoid(self.fc2(x))  # Sigmoid
        return swish_part * gate 
    

class MLP(nn.Module):
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool, activation: str,
        dropout_prob: float, norm_type: str = "batch", NEW_BATCH_NORM=False, use_bias=True
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, use_bn, activation, dropout_prob, norm_type, NEW_BATCH_NORM)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=use_bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X, mask=mask)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features


class MLPLayer(nn.Module):
    """
    Based on https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """
    fc: nn.Linear
    bn: Optional[nn.BatchNorm1d]
    activation: nn.Module
    dropout: nn.Dropout

    def __init__(self, in_dims: int, out_dims: int, use_bn: bool, activation: str,
                 dropout_prob: float, norm_type: str = "batch", NEW_BATCH_NORM=False) -> None:
        super().__init__()
        # self.fc = nn.Linear(in_dims, out_dims, bias=not use_bn)
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        self.NEW_BATCH_NORM = NEW_BATCH_NORM
        if NEW_BATCH_NORM:
            self.bn = nn.BatchNorm1d(out_dims)
        elif use_bn:
            self.bn = nn.BatchNorm1d(out_dims) if norm_type == "batch" else nn.LayerNorm(out_dims)
        else:
            self.bn = None
        # self.bn = nn.BatchNorm1d(out_dims) if use_bn else None
        # self.ln = nn.LayerNorm(out_dims) if use_bn else None

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'none':
            self.activation = nn.Identity()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'swiglu':
            self.activation = SwiGLU(out_dims)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation!")
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        X = self.fc(X)                     # [***, D_out]
        if mask is not None:
            X[~mask] = 0
        if self.NEW_BATCH_NORM:
            if mask is None:
                X = X.transpose(0, 1)
                X = self.bn(X.transpose(1, 2)).transpose(1,2)
                X = X.transpose(0, 1)
            else:
                X[mask] = self.bn(X[mask])
        elif self.bn is not None:
            shape = X.size()
            X = X.reshape(-1, shape[-1])   # [prod(***), D_out]
            X = self.bn(X)                 # [prod(***), D_out]
            X = X.reshape(shape)           # [***, D_out]
        X = self.activation(X)             # [***, D_out]
        X = self.dropout(X)                # [***, D_out]
        return X
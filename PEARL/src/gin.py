from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj , degree
from src.mlp import MLP

class GIN(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False, laplacian=None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims, hidden_dims))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
        layer = GINLayer(create_mlp(hidden_dims, out_dims))
        self.layers.append(layer)
        self.laplacian=laplacian
        print("GINPHI LAP is: ", laplacian)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index, laplacian=self.laplacian, mask=mask)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            if mask is not None:
                X[~mask] = 0
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if mask is None:
                    if X.ndim == 3:
                        X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                    else:
                        X = self.batch_norms[i](X)
                else:
                    X[mask] = self.batch_norms[i](X[mask])
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True) #torch.empty(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, laplacian=False, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        if laplacian == 'L':
            edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=X.size(0))
            laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', laplacian, X)   # [N_sum, ***, D_in]

            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z, mask)  
        elif laplacian == 'RW':
            adj = to_dense_adj(edge_index).squeeze(0)  # [N_sum, N_sum]
            deg = degree(edge_index[0], num_nodes=X.size(0), dtype=torch.float)  # [N_sum]
            deg_inv = 1.0 / deg  # Inverse of the degree
            deg_inv[deg_inv == float('inf')] = 0  # Handle division by zero for isolated nodes
            deg_inv_diag = torch.diag(deg_inv)  # [N_sum, N_sum]
            random_walk = torch.matmul(adj, deg_inv_diag)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', random_walk, X)  # [N_sum, *, D_in]
            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z, mask)       # [N_sum, ***, D_out]

        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z, mask)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims

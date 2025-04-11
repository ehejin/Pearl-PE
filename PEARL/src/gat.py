import torch
from typing import Callable
from torch import nn
from torch_geometric.nn import MessagePassing
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from src.mlp import MLP
from torch_scatter import scatter_softmax

class GATLayer(MessagePassing):
    def __init__(self, n_edge_types: int, in_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
                 feature_type: str = "discrete", pe_emb=37, heads=4, concat=True, dropout=0.0):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.in_dims = in_dims
        self.out_dims = out_dims // heads if concat else out_dims

        self.edge_features = nn.Embedding(n_edge_types + 1, in_dims) if feature_type == "discrete" else \
                             nn.Linear(n_edge_types, in_dims)
        self.pe_embedding = create_mlp(pe_emb, in_dims)  # For PE-full
        self.attn_l = nn.Parameter(torch.Tensor(heads, self.out_dims))
        self.attn_r = nn.Parameter(torch.Tensor(heads, self.out_dims))
        self.attn_edge = nn.Parameter(torch.Tensor(heads, self.out_dims)) if feature_type == "discrete" else None
        self.lin = nn.Linear(in_dims, heads * self.out_dims, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_dims))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.attn_l)
        glorot(self.attn_r)
        if self.attn_edge is not None:
            glorot(self.attn_edge)
        zeros(self.bias)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :param PE: Positional encoding. [N_sum, pe_emb]
        :return: Output node feature matrix. [N_sum, D_out]
        """
        if PE is not None:
            X_n = X_n + self.pe_embedding(PE)

        X_n = self.lin(X_n).view(-1, self.heads, self.out_dims)  # [N_sum, heads, D_out]

        X_e = self.edge_features(edge_attr) if edge_attr is not None else None
        if X_e is not None and self.attn_edge is not None:
            X_e = torch.einsum("eh,eh->eh", X_e, self.attn_edge)  # Scale edge features

        return self.propagate(edge_index, X=X_n, edge_attr=X_e) + self.bias

    def message(self, X_j: torch.Tensor, X_i: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        alpha = (X_j * self.attn_l).sum(dim=-1) + (X_i * self.attn_r).sum(dim=-1)
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        if edge_attr is not None:
            alpha += (edge_attr).sum(dim=-1)

        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Compute softmax per target node
        alpha = scatter_softmax(alpha, index, dim=0)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * X_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_dims)  # Concatenate heads
        else:
            return aggr_out.mean(dim=1)  # Average heads

class GAT(nn.Module):
    def __init__(self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP], heads: int = 1, bn: bool = True, residual: bool = True,
                 feature_type: str = "discrete", pe_emb=37, dropout=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        self.bn = bn
        self.heads = heads

        if bn:
            self.batch_norms = nn.ModuleList()

        for _ in range(n_layers - 1):
            layer = GATLayer(n_edge_types, in_dims, hidden_dims, create_mlp, feature_type, pe_emb, heads, True, dropout)
            self.layers.append(layer)
            in_dims = hidden_dims * heads if True else hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(in_dims))

        layer = GATLayer(n_edge_types, in_dims, out_dims, create_mlp, feature_type, pe_emb, heads, False, dropout)
        self.layers.append(layer)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            X_0 = X_n
            X_n = layer(X_n, edge_index, edge_attr, PE)  # [N_sum, D_hid] or [N_sum, D_out]
            if self.bn and i < len(self.layers) - 1:
                if X_n.ndim == 3:
                    X_n = self.batch_norms[i](X_n.transpose(2, 1)).transpose(2, 1)
                else:
                    X_n = self.batch_norms[i](X_n)
            if self.residual:
                X_n = X_n + X_0
        return X_n  # [N_sum, D_out]
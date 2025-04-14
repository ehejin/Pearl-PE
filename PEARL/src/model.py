from typing import Callable

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

from src.gine import GINE
from src.mlp import MLP
from src.pe import PEARLPositionalEncoder, GetSampleAggregator
from src.gin import GIN
from src.schema import Schema
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import BatchNorm, global_mean_pool

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/model.py 

def construct_model(cfg: Schema, list_create_mlp, rgnn=True, deg1=None, **kwargs): 
    create_mlp, create_mlp_ln = list_create_mlp if isinstance(list_create_mlp, tuple) else (list_create_mlp, list_create_mlp)
    target_dim = cfg.target_dim
    if cfg.base_model == 'gine':
        base_model = GINEBaseModel(
            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, create_mlp,
            residual=kwargs.get("residual"), feature_type=kwargs.get('feature_type'), pooling=cfg.pooling,
            target_dim=target_dim, bn=cfg.gine_model_bn, pe_emb=cfg.pe_dims
        )
    else:
        raise Exception("Base model not implemented!")
    if cfg.pe_method == 'pearl':
        Phi = GetSampleAggregator(cfg, create_mlp_ln, kwargs['device']) 
        pe_model = PEARLPositionalEncoder(Phi, cfg.BASIS, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, mlp_hid=cfg.RAND_mlp_hid, pearl_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)
        return PEARL_GNN_Model(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=pe_model,
            base_model=base_model,
            pe_aggregate = cfg.pe_aggregate,
            feature_type = kwargs.get("feature_type")
        )
    else:
        raise Exception("PE method not implemented!")

'''
    This is our GNN with PEARL PE processing. We pass in the laplacian along with W, our random/basis vectors.
'''
class PEARL_GNN_Model(nn.Module):
    node_features: nn.Embedding
    positional_encoding: nn.Module
    fc: nn.Linear
    base_model: nn.Module

    def __init__(
        self, n_node_types: int, node_emb_dims: int, positional_encoding: nn.Module, base_model: nn.Module,
            pe_aggregate: str, feature_type: str = "discrete"
    ) -> None:
        super().__init__()
        self.node_features = nn.Embedding(n_node_types, node_emb_dims) if feature_type == "discrete" else nn.Linear(n_node_types, node_emb_dims) #
        self.base_model = base_model
        self.positional_encoding = positional_encoding
        if positional_encoding is not None:
            self.pe_embedding = nn.Linear(self.positional_encoding.out_dims, node_emb_dims)
            self.pe_aggregate = pe_aggregate 
            assert pe_aggregate == "add" or pe_aggregate == "concat" or pe_aggregate == "peg"
            if pe_aggregate == "concat":
                self.fc = nn.Linear(2 * node_emb_dims, node_emb_dims, bias=True)

    def forward(self, batch: Batch, W) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1)) 
        PE = None
        if self.positional_encoding is not None:
            PE = self.positional_encoding(batch.Lap, W, batch.edge_index, batch.batch)  
            if self.pe_aggregate == "add":
                X_n = X_n + self.pe_embedding(PE)
            elif self.pe_aggregate == "concat":
                X_n = torch.cat([X_n, self.pe_embedding(PE)], dim=-1)
                X_n = self.fc(X_n)                                                                  
            elif self.pe_aggregate == "peg":
                PE = torch.linalg.norm(PE[batch.edge_index[0]] - PE[batch.edge_index[1]], dim=-1)
                PE = PE.view([-1, 1])
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.snorm if "snorm" in batch else None
                               , batch.batch)  

class GINEBaseModel(nn.Module):
    gine: GINE

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, create_mlp: Callable[[int, int], MLP],
            residual: bool = False, bn: bool = False, feature_type: str = "discrete", pooling: str = "mean",
            target_dim: int = 1, pe_emb=37) -> None:
        super().__init__()
        print("GINEBase BN is: ", bn)
        self.gine = GINE(n_layers, n_edge_types, in_dims, hidden_dims, hidden_dims, create_mlp, residual=residual,
                         bn=bn, feature_type=feature_type, pe_emb=pe_emb)
        self.mlp = create_mlp(hidden_dims, target_dim)
        self.pooling = global_mean_pool if pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.gine(X_n, edge_index, edge_attr, PE)  
        X_n = self.pooling(X_n, batch) 
        Y_pred = self.mlp(X_n)        
        return Y_pred.squeeze(dim=1)  

 

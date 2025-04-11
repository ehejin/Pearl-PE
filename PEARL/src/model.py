from typing import Callable

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

from src.gine import GINE
from src.deepsets import DeepSets
from src.mlp import MLP
from src.pe import PEARL_PE, GetPhi, GetPsi
from src.sign_inv_pe import SignInvPe_PEARL, MaskedSignInvPe_PEARL
from src.vanilla_pe import IdPE
from src.gin import GIN
from src.pna import PNA
from src.schema import Schema
from src.gated_gcn import GatedGCNLayer
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import BatchNorm, global_mean_pool
from src.gat import GAT
from src.gated_gcn import GATEDGNN


def construct_model(cfg: Schema, list_create_mlp, rgnn=True, deg1=None, **kwargs): # a list of mlp creators
    create_mlp, create_mlp_ln = list_create_mlp if isinstance(list_create_mlp, tuple) else (list_create_mlp, list_create_mlp)
    target_dim = cfg.target_dim# if kwargs.get('target_dim') is None else kwargs.get('target_dim')
    if cfg.base_model == 'gine':
        base_model = GINEBaseModel(
            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, create_mlp,
            residual=kwargs.get("residual"), feature_type=kwargs.get('feature_type'), pooling=cfg.pooling,
            target_dim=target_dim, bn=cfg.gine_model_bn, pe_emb=cfg.pe_dims
        )
    elif cfg.base_model == 'pna':
        # cannot use now
        assert kwargs.get("deg") is not None
        base_model = PNABaseModel(cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims,
                                  create_mlp, deg=kwargs.get("deg"), residual=kwargs.get("residual"),
                                  bn=kwargs.get("bn"), sn=kwargs.get("sn"))
    elif cfg.base_model == 'gated_gcn':
        base_model = Gated_GCNBaseModel(
            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, create_mlp,
            residual=kwargs.get("residual"), feature_type=kwargs.get('feature_type'), pooling=cfg.pooling,
            target_dim=target_dim, bn=cfg.gine_model_bn, pe_emb=cfg.pe_dims
        )
    elif cfg.base_model == 'gat':
        base_model = GATBaseModel(
            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, create_mlp,
            residual=kwargs.get("residual"), feature_type=kwargs.get('feature_type'), pooling=cfg.pooling,
            target_dim=target_dim, bn=cfg.gine_model_bn, pe_emb=cfg.pe_dims
        )
    elif cfg.base_model == 'pna_molhiv':
        base_model = Net(deg=deg1, create_mlp=create_mlp)
        #return base_model
        #base_model = GatedGCNLayer(in_dim, out_dim)
    else:
        raise Exception("Base model not implemented!")
    # pe_method
    if cfg.pe_method == "sign_inv":
        if not rgnn:
            gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, 4, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
            rho = MLP(cfg.n_psi_layers, cfg.pe_dims * 4, cfg.phi_hidden_dims,
                    cfg.pe_dims, use_bn=cfg.mlp_use_bn, activation='relu', dropout_prob=0.0)
            return Model(
                cfg.n_node_types, cfg.node_emb_dims,
                positional_encoding=SignInvPe_PEARL(phi=gin, rho=rho),
                base_model=base_model,
                pe_aggregate=cfg.pe_aggregate,
                feature_type=kwargs.get("feature_type")
            )
        else:
            gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, 30, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
            rho = MLP(cfg.n_psi_layers, 30, cfg.phi_hidden_dims,
                    cfg.pe_dims, use_bn=cfg.mlp_use_bn, activation='relu', dropout_prob=0.0)
            return Model_R(
                cfg.n_node_types, cfg.node_emb_dims,
                positional_encoding=SignInvPe_PEARL(phi=gin, rho=rho),
                base_model=base_model,
                pe_aggregate=cfg.pe_aggregate,
                feature_type=kwargs.get("feature_type")
            )
    elif cfg.pe_method == 'masked_sign_inv':
        if not rgnn:
            gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, cfg.phi_hidden_dims, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
            # rho = create_mlp(cfg.phi_hidden_dims, cfg.pe_dims)
            rho = MLP(cfg.n_psi_layers, cfg.phi_hidden_dims, cfg.phi_hidden_dims, cfg.pe_dims, use_bn=cfg.mlp_use_bn,
                    activation='relu', dropout_prob=0.0)
            return Model(
                cfg.n_node_types, cfg.node_emb_dims,
                positional_encoding=MaskedSignInvPe_PEARL(phi=gin, rho=rho),
                base_model=base_model,
                pe_aggregate=cfg.pe_aggregate,
                feature_type=kwargs.get("feature_type")
            )
        else:
            gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, cfg.phi_hidden_dims, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
            # rho = create_mlp(cfg.phi_hidden_dims, cfg.pe_dims)
            rho = MLP(cfg.n_psi_layers, cfg.phi_hidden_dims, cfg.phi_hidden_dims, cfg.pe_dims, use_bn=cfg.mlp_use_bn,
                    activation='relu', dropout_prob=0.0)
            return Model_R(
                cfg.n_node_types, cfg.node_emb_dims,
                positional_encoding=MaskedSignInvPe_PEARL(phi=gin, rho=rho),
                base_model=base_model,
                pe_aggregate=cfg.pe_aggregate,
                feature_type=kwargs.get("feature_type")
            )
    elif cfg.pe_method.endswith('spe'):
        Phi = GetPhi(cfg, create_mlp_ln, kwargs['device']) # for phi function, use layer norm
        Psi_list = [
            GetPsi(cfg)
            for _ in range(cfg.n_psis)
        ]
        if cfg.pe_method == 'spe':
            pe_model = PEARL_PE(Phi, Psi_list, cfg.BASIS, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)
        return Model_GNN_PEARL(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=pe_model,
            base_model=base_model,
            pe_aggregate = cfg.pe_aggregate,
            feature_type = kwargs.get("feature_type")
        )
    elif cfg.pe_method == 'none':
        return Model(cfg.n_node_types, cfg.node_emb_dims, positional_encoding=None, base_model=base_model,
                     pe_aggregate=None, feature_type = kwargs.get("feature_type"))
    elif cfg.pe_method == 'id':
        return Model(cfg.n_node_types, cfg.node_emb_dims, positional_encoding=IdPE(cfg.pe_dims), base_model=base_model,
                     pe_aggregate=cfg.pe_aggregate, feature_type = kwargs.get("feature_type"))
    else:
        raise Exception("PE method not implemented!")


class Model(nn.Module):
    node_features: nn.Embedding
    positional_encoding: nn.Module
    fc: nn.Linear
    base_model: nn.Module

    def __init__(
        self, n_node_types: int, node_emb_dims: int, positional_encoding: nn.Module, base_model: nn.Module,
            pe_aggregate: str, feature_type: str = "discrete"
    ) -> None:
        super().__init__()

        self.node_features = nn.Embedding(n_node_types, node_emb_dims) if feature_type == "discrete" else \
                             nn.Linear(n_node_types, node_emb_dims)
        self.base_model = base_model
        self.positional_encoding = positional_encoding
        if positional_encoding is not None:
            self.pe_embedding = nn.Linear(self.positional_encoding.out_dims, node_emb_dims)
            self.pe_aggregate = pe_aggregate # "add" or "concat"
            assert pe_aggregate == "add" or pe_aggregate == "concat" or pe_aggregate == "peg"
            if pe_aggregate == "concat":
                self.fc = nn.Linear(2 * node_emb_dims, node_emb_dims, bias=True)
        # self.fc = nn.Linear(self.positional_encoding.out_dims, node_emb_dims, bias=True)

    def forward(self, batch: Batch) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1))    # [N_sum, D]
        PE = None
        if self.positional_encoding is not None:
            eig_mats = batch.P if "P" in batch else batch.V # pass projection matrices if using BasisNet
            PE = self.positional_encoding(batch.Lambda, eig_mats, batch.edge_index, batch.batch)   # [N_sum, D_pe]
            if self.pe_aggregate == "add":
                X_n = X_n + self.pe_embedding(PE)
            elif self.pe_aggregate == "concat":
                # PE = self.pe_embedding(PE)
                X_n = torch.cat([X_n, self.pe_embedding(PE)], dim=-1)
                X_n = self.fc(X_n)                                                                    # [N_sum, D]
            elif self.pe_aggregate == "peg":
                PE = torch.linalg.norm(PE[batch.edge_index[0]] - PE[batch.edge_index[1]], dim=-1)
                PE = PE.view([-1, 1])
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.snorm if "snorm" in batch else None
                               , batch.batch)           # [B]

class Model_R(nn.Module):
    node_features: nn.Embedding
    positional_encoding: nn.Module
    fc: nn.Linear
    base_model: nn.Module

    def __init__(
        self, n_node_types: int, node_emb_dims: int, positional_encoding: nn.Module, base_model: nn.Module,
            pe_aggregate: str, feature_type: str = "discrete"
    ) -> None:
        super().__init__()

        self.node_features = nn.Embedding(n_node_types, node_emb_dims) if feature_type == "discrete" else \
                             nn.Linear(n_node_types, node_emb_dims)
        self.base_model = base_model
        self.positional_encoding = positional_encoding
        if positional_encoding is not None:
            self.pe_embedding = nn.Linear(self.positional_encoding.out_dims, node_emb_dims)
            self.pe_aggregate = pe_aggregate # "add" or "concat"
            assert pe_aggregate == "add" or pe_aggregate == "concat" or pe_aggregate == "peg"
            if pe_aggregate == "concat":
                self.fc = nn.Linear(2 * node_emb_dims, node_emb_dims, bias=True)
        # self.fc = nn.Linear(self.positional_encoding.out_dims, node_emb_dims, bias=True)

    def forward(self, batch: Batch, r) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1))    # [N_sum, D]
        PE = None
        if self.positional_encoding is not None:
            PE = self.positional_encoding(None, r, batch.edge_index, batch.batch)   # [N_sum, D_pe]
            if self.pe_aggregate == "add":
                X_n = X_n + self.pe_embedding(PE)
            elif self.pe_aggregate == "concat":
                X_n = torch.cat([X_n, self.pe_embedding(PE)], dim=-1)
                X_n = self.fc(X_n)                                                                    # [N_sum, D]
            elif self.pe_aggregate == "peg":
                PE = torch.linalg.norm(PE[batch.edge_index[0]] - PE[batch.edge_index[1]], dim=-1)
                PE = PE.view([-1, 1])
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.snorm if "snorm" in batch else None
                               , batch.batch) 
    
'''
    This is our GNN with PEARL PE processing. We pass in the laplacian along with W, our random/basis vectors.
'''
class Model_GNN_PEARL(nn.Module):
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
            self.pe_aggregate = pe_aggregate # "add" or "concat"
            assert pe_aggregate == "add" or pe_aggregate == "concat" or pe_aggregate == "peg"
            if pe_aggregate == "concat":
                self.fc = nn.Linear(2 * node_emb_dims, node_emb_dims, bias=True)

    def forward(self, batch: Batch, W) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1))    # [N_sum, D]
        PE = None
        if self.positional_encoding is not None:
            PE = self.positional_encoding(batch.Lap, W, batch.edge_index, batch.batch)   # [N_sum, D_pe]
            if self.pe_aggregate == "add":
                X_n = X_n + self.pe_embedding(PE)
                # PE = None
            elif self.pe_aggregate == "concat":
                X_n = torch.cat([X_n, self.pe_embedding(PE)], dim=-1)
                X_n = self.fc(X_n)                                                                    # [N_sum, D]
            elif self.pe_aggregate == "peg":
                PE = torch.linalg.norm(PE[batch.edge_index[0]] - PE[batch.edge_index[1]], dim=-1)
                PE = PE.view([-1, 1])
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.snorm if "snorm" in batch else None
                               , batch.batch)  

class Net(torch.nn.Module):
    def __init__(self, deg=None, create_mlp=None):
        super(Net, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConvSimple(in_channels=80, out_channels=80, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1, create_mlp=create_mlp, pe_embedding=37)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))

        self.mlp = Sequential(Linear(80, 40), ReLU(), Linear(40, 20), ReLU(), Linear(20, 1))

    def forward(self, x, edge_index, edge_attr, PE, snorm, batch):#x, edge_index, edge_attr, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, PE)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class GINEBaseModel(nn.Module):
    gine: GINE

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, create_mlp: Callable[[int, int], MLP],
            residual: bool = False, bn: bool = False, feature_type: str = "discrete", pooling: str = "mean",
            target_dim: int = 1, pe_emb=37) -> None:
        super().__init__()
        print("ARE WE USING GINEBASE BN?:", bn)
        self.gine = GINE(n_layers, n_edge_types, in_dims, hidden_dims, hidden_dims, create_mlp, residual=residual,
                         bn=bn, feature_type=feature_type, pe_emb=pe_emb)
        self.mlp = create_mlp(hidden_dims, target_dim)
        self.pooling = global_mean_pool if pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.gine(X_n, edge_index, edge_attr, PE)   # [N_sum, hidden_dim]
        X_n = self.pooling(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1)          


class PNABaseModel(nn.Module):
    def __init__(self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int,
                 create_mlp: Callable[[int, int], MLP], deg: torch.Tensor, residual: bool = False, bn: bool = False,
                 sn: bool = False
                 ) -> None:
        super(PNABaseModel, self).__init__()
        self.pna = PNA(n_layers, n_edge_types, in_dims, hidden_dims, create_mlp, deg=deg, residual=residual,
                       bn=bn, sn=sn)
        self.mlp = create_mlp(hidden_dims, 1)

    def forward(
            self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.pna(X_n, edge_index, edge_attr, snorm)   # [N_sum, hidden_dim]
        X_n = global_add_pool(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1)                  # [B]


class Gated_GCNBaseModel(nn.Module):

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, create_mlp: Callable[[int, int], MLP],
            residual: bool = False, bn: bool = False, feature_type: str = "discrete", pooling: str = "mean",
            target_dim: int = 1, pe_emb=37) -> None:
        super().__init__()
        print("ARE WE USING BASE BN?:", bn)
        self.gine = GATEDGNN(n_layers, n_edge_types, in_dims, hidden_dims, hidden_dims, create_mlp, residual=residual,
                         bn=bn, feature_type=feature_type, pe_emb=pe_emb)
        self.mlp = MLP(
            3, hidden_dims, 70, target_dim, True,
            'relu', 0.0
         )
        #self.mlp = create_mlp(hidden_dims, target_dim)
        self.pooling = global_mean_pool if pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.gine(X_n, edge_index, edge_attr, PE)   # [N_sum, hidden_dim]
        X_n = self.pooling(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1)  

class GATBaseModel(nn.Module):
    gine: GINE

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, create_mlp: Callable[[int, int], MLP],
            residual: bool = False, bn: bool = False, feature_type: str = "discrete", pooling: str = "mean",
            target_dim: int = 1, pe_emb=37) -> None:
        super().__init__()
        self.gat = GAT(n_layers, n_edge_types, in_dims, hidden_dims, hidden_dims, create_mlp, residual=residual,
                         bn=bn, feature_type=feature_type, pe_emb=pe_emb)
        self.mlp = create_mlp(hidden_dims, target_dim)
        self.pooling = global_mean_pool if pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.gat(X_n, edge_index, edge_attr, PE)   # [N_sum, hidden_dim]
        # X_n = global_mean_pool(X_n, batch) # [B, hidden_dim]
        X_n = self.pooling(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1) 
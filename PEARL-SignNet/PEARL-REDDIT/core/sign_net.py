import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from core.transform import to_dense_list_EVD

import core.model_utils.masked_layers as masked_layers 
import core.model_utils.pyg_gnn_wrapper as elements
from core.model import GNN
from core.model_utils.transformer_module import TransformerEncoderLayer, PositionalEncoding 
from core.model_utils.elements import DiscreteEncoder

class GNN3d(nn.Module): 
    """
    Apply GNN on a 3-dimensional data x: n x k x d. 
    Equivalent to apply GNN on k independent nxd 2-d feature.
    * Assume no edge feature for now.
    """
    def __init__(self, n_in, n_out, n_layer, gnn_type='MaskedGINConv', NOUT=128, skipc=True):
        super().__init__()
        self.convs = nn.ModuleList([getattr(masked_layers, gnn_type)(n_in if i==0 else n_out, n_out, bias=False) for i in range(n_layer)])
        self.norms = nn.ModuleList([masked_layers.MaskedBN(n_out) for _ in range(n_layer)])
        self.skipc = skipc
        if skipc:
            self.output_encoder = nn.Linear(n_layer*n_out, n_out,bias=False)
            self.output_norm = masked_layers.MaskedBN(n_out)

    def reset_parameters(self):
        if self.skipc:
            self.output_encoder.reset_parameters()
            self.output_norm.reset_parameters()
        for conv, norm in zip(self.convs, self.norms): 
            conv.reset_parameters()
            norm.reset_parameters() 
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: n x k x d
        # mask: n x k
        x = x.transpose(0, 1) # k x n x d
        if mask is not None:
            mask = mask.transpose(0, 1) # k x n
        previous_x = 0
        skip_connections = []
        for conv, norm in zip(self.convs, self.norms):
            #TODO: current not work for continuous edge attri
            # x = conv(x, edge_index, enc(edge_attr), mask) # pass mask into
            x = conv(x, edge_index, edge_attr, mask) # pass mask into
            # assert x[~mask].max() == 0 
            if mask is not None:
                x[~mask] = 0
            x = norm(x, mask)
            x = F.relu(x)
            if self.skipc:
                skip_connections.append(x)
            else:
                x = x + previous_x
                previous_x = x
        if self.skipc:
            x = torch.cat(skip_connections, dim=-1)
            x = self.output_encoder(x)
            x = self.output_norm(x) # maybe play w this
        return x.transpose(0, 1)

'''
    This is our PEARL model that builds off of SignNet. We simply pass in basis or random vectors into
    the GNN3D model and take the expectation.
'''
class PEARL(nn.Module): 
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2, NOUT=128):
        super().__init__()
        self.phi = GNN3d(1, n_hid, nl_phi, gnn_type='MaskedGINConv', NOUT=NOUT) 

    def reset_parameters(self):
        self.phi.reset_parameters()

    def forward(self, data, rand):
        x = rand
        mask_full = None
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) 
        x = x.mean(dim=1)
        return x
    

class PE_GNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid_3d, n_hid, n_out, nl_pe_gnn, nl_gnn):
        super().__init__()
        self.n_out = n_out
        self.pe_gnn = PEARL(n_hid_3d, nl_pe_gnn, nl_rho=1, NOUT=n_hid)
        self.is_hid_neq = (n_hid != n_hid_3d)
        if self.is_hid_neq:
            self.out = nn.Sequential(nn.Linear(n_hid_3d, n_hid, bias=False), nn.BatchNorm1d(n_hid))
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type='GINConv') #GINEConv

    def reset_parameters(self):
        if self.is_hid_neq:
            for layer in self.out:
                layer.reset_parameters()
        self.pe_gnn.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data, r):
        pos = self.pe_gnn(data, r)
        if self.is_hid_neq:
            for layer in self.out:
                pos = layer(pos)
        return self.gnn(data, pos)
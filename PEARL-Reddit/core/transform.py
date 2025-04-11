# Transform for extracting eigenvector and eigenvalues 
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian, to_undirected

# The needed pretransform to save result of EVD
class EVDTransform(object): 
    def __init__(self, norm=None):
        super().__init__()
        self.norm = norm
    def __call__(self, data):
        D, V = EVD_Laplacian(data, self.norm)
        data.eigen_values = D
        data.eigen_vectors = V.reshape(-1) # reshape to 1-d to save 
        return data

def EVD_Laplacian(data, norm=None):
    L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
                          normalization=norm, num_nodes=data.num_nodes)
    L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

    D, V  = torch.linalg.eigh(L)
    return D, V



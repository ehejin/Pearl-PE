import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
from core.config import cfg, update_cfg
from core.train import run
from core.model import GNN
from core.pe import PE_GNN
from core.transform import EVDTransform
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

from torch_geometric.datasets import ZINC, TUDataset

def check_distinct(data):
    return len(data.eigen_values) == len(torch.unique(data.eigen_values)) 

def stratified_split(dataset, seed, fold_idx, n_splits=10):
    assert 0 <= fold_idx < n_splits, "fold_idx must be from 0 to n_splits-1."
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [data.y.item() for data in dataset]
    
    idx_list = []
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append((train_idx, test_idx))
    
    train_idx, test_idx = idx_list[fold_idx]
    
    # Create train and test datasets based on the indices
    train_dataset = dataset[torch.tensor(train_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]
    
    return train_dataset, test_dataset

def create_dataset(cfg, fold_idx): 
    torch.set_num_threads(cfg.num_workers)
    #transform = transform_eval = EVDTransform('sym')
    transform = transform_eval = None
    if cfg.dataset == 'ZINC':
        root = 'data/ZINC'
        train_dataset = ZINC(root, subset=True, split='train', transform=transform)
        val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
        test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval)   
    else:
        root = 'data/' + cfg.dataset
        dataset = TUDataset(root, name=cfg.dataset, transform=transform)
        labels = [data.y.item() for data in dataset]
        seed = 42  # Set your seed
        train_dataset, test_dataset = stratified_split(dataset, seed, fold_idx)
        val_dataset = None
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    if cfg.model.gnn_type == 'PEARL':
        model = PE_GNN(None, None,
                           n_hid_3d=cfg.model.hidden_size_3d, 
                           n_hid = cfg.model.hidden_size,
                           n_out=cfg.model.n_out, 
                           nl_pe_gnn=cfg.model.num_layers_pe, 
                           nl_gnn=cfg.model.num_layers)
    else:
        model = GNN(None, None, 
                    nhid=cfg.model.hidden_size, 
                    nout=1, 
                    nlayer=cfg.model.num_layers, 
                    gnn_type=cfg.model.gnn_type, 
                    dropout=cfg.train.dropout, 
                    pooling=cfg.model.pool,
                    res=True)

    return model


def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0 
    for data in train_loader:
        r = 1+torch.randn(data.num_nodes, 80, 1).to(torch.device('cuda')) #NxMx1
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = (model(data, r).squeeze() - y).abs().mean()
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0
    for data in loader:
        r = 1+torch.randn(data.num_nodes, 80, 1).to(torch.device('cuda'))
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data, r).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf

def train_REDDIT(train_loader, model, optimizer, device, samples=30):
    total_loss = 0
    N = 0 
    if model.n_out == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        r = 1+torch.randn(data.num_nodes, samples, 1).to(torch.device('cuda')) #NxMx1
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        data.x = torch.ones(data.num_nodes,1).long().to(device)
        optimizer.zero_grad()
        out = model(data, r).squeeze()
        #out = model(data).squeeze()
        loss = criterion(out.float(), y.float()) # float if binary, long if multi    
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test_REDDIT(loader, model, evaluator, device, samples=30):
    N = 0
    total = 0
    correct = 0
    for data in loader:
        r = 1+torch.randn(data.num_nodes, samples, 1).to(torch.device('cuda'))
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        data.x = torch.ones(data.num_nodes,1).long().to(device)
        if model.n_out == 1:
            outputs = torch.sigmoid(model(data, r)).squeeze()
            #outputs = torch.sigmoid(model(data)).squeeze()
            predicted = (outputs > 0.5).squeeze()  
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)
        else:
            outputs = model(data, r)
            #outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item() 
            total += y.size(0)
    return correct / total


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser(description="give_config")
    #parser.add_argument('--num_samples1', type=int, default=0, action='store', required=True)
    parser.add_argument('--config', type=str, default='zinc.yaml', help="Path to the config file")
    args = parser.parse_args()
    print(args)
    cfg.merge_from_file('train/config/'+args.config)
    cfg = update_cfg(cfg)
    if cfg.dataset == 'ZINC':
        run(cfg, create_dataset, create_model, train, test)
    else:
        run(cfg, create_dataset, create_model, train_REDDIT, test_REDDIT, cfg.num_samples)

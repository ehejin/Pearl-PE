from dataclasses import dataclass

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/schema.py 

@dataclass
class Schema:
    # model attributes
    base_model: str
    pe_method: str
    n_node_types: int
    n_edge_types: int
    node_emb_dims: int
    pooling: str

    sample_aggr_model_name: str
    pe_dims: int
    n_sample_aggr_layers: int
    sample_aggr_hidden_dims: int

    n_base_layers: int
    base_hidden_dims: int

    n_mlp_layers: int
    mlp_hidden_dims: int
    mlp_use_bn: bool
    mlp_use_ln: bool
    mlp_activation: str
    mlp_dropout_prob: float
    pe_aggregate: str

    residual: bool
    batch_norm: bool
    graph_norm: bool

    # data attributes
    use_subset: bool
    train_batch_size: int
    val_batch_size: int
    # class_weight: bool

    # optimizer attributes
    lr: float
    weight_decay: float
    momentum: float
    nesterov: bool

    # scheduler attributes
    n_warmup_steps: int

    # miscellaneous
    n_epochs: int
    out_dirpath: str

    # PEARL PE model attributes
    basis: bool    # If TRUE will run using basis vectors, otherwise will use random samples
    num_samples: int  # number of random samples (only used if basis is false)
    pearl_k: int  # number of graph filters to apply
    pearl_mlp_nlayers: int  # number of mlp layers after graph filtering and before the sample aggr
    pearl_mlp_hid: int # hidden size of mlp after graph filtering
    wandb: bool
    wandb_run_name: str
    pearl_act: str # activation of mlp after graph filtering
    pearl_mlp_out: int # output size of mlp after graph filters
    
    gine_model_bn: bool
    target_dim: int
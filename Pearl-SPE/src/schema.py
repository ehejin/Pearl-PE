from dataclasses import dataclass


@dataclass
class Schema:
    # model attributes
    base_model: str
    pe_method: str
    n_node_types: int
    n_edge_types: int
    node_emb_dims: int
    pooling: str


    phi_model_name: str
    pe_dims: int
    n_phi_layers: int
    phi_hidden_dims: int
    psi_model_name: str
    n_psis: int
    n_psi_layers: int
    psi_hidden_dims: int
    psi_activation: str
    num_heads: int # for transformers
    pe_aggregate: str

    n_base_layers: int
    base_hidden_dims: int

    n_mlp_layers: int
    mlp_hidden_dims: int
    mlp_use_bn: bool
    mlp_use_ln: bool
    mlp_activation: str
    mlp_dropout_prob: float

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
    BASIS: bool    # IF TRUE will run using basis vectors, otherwise will use random samples
    num_samples: int
    RAND_k: int
    RAND_mlp_nlayers: int
    RAND_mlp_hid: int
    wandb: bool
    wandb_run_name: str
    RAND_act: str
    RAND_LAP: str
    RAND_mlp_out: int
    gine_model_bn: bool
    target_dim: int
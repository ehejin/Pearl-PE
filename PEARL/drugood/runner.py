import sys
sys.path.append('../') # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.schema import Schema
from trainer import Trainer
import wandb
from root import root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)

    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size

    if cfg.wandb:
        wandb.login(key="") # use your own Wandb key
        wandb.init(dir=root("."), project="PEARL-drugood", name=cfg.wandb_run_name, config=cfg.__dict__,
                   settings=wandb.Settings(code_dir="."))

    for i in [0, 1, 2, 4, 42]:
        cfg.seed = i
        cfg.dataset = args.dataset
        trainer = Trainer(cfg, args.gpu_id)
        trainer.train()

if __name__ == "__main__":
    main()
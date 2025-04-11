import torch
import time
from core.log import config_logger
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
import matplotlib.pyplot as plt
import csv

def run(cfg, create_dataset, create_model, train, test, evaluator=None, samples=80, test_samples=200, runs=''):
    set_random_seed(seed=42)
    writer, logger, config_string = config_logger(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg, 0)

    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    test_perfs = []
    vali_perfs = []
    run_tests = []
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,  # Replace with your project name
            name=cfg.run_name,         # Optional: Give your run a name
            config=cfg
        )
    start_str, stop_str = runs.split("-")
    start = int(start_str)
    stop = int(stop_str)
    all_test_curves = []
    for run in range(start, stop): 
        if run != 1:
            train_dataset, val_dataset, test_dataset = create_dataset(cfg, run-1)
        model = create_model(cfg).to(torch.device(cfg.device))
        # print(f"Number of parameters: {count_parameters(model)}")
        model.reset_parameters()
        print("TOTAL PARAMS", sum(p.numel() for p in model.parameters()))
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

        # 4. train
        start_outer = time.time()
        best_val_perf = test_perf = float('-inf')
        maximum_test = 0
        test_curve = []
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            model.train()
            train_loss = train(train_loader, model, optimizer, device=cfg.device, samples=samples)
            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)
            model.eval()
            val_perf = 0
            test_perf = test(test_loader, model, evaluator=evaluator, device=cfg.device, test_samples=test_samples)
            test_curve.append(test_perf)
            time_per_epoch = time.time() - start 

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
            if cfg.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "Val": val_perf,
                    "Test": test_perf
                })
            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/test-best-perf', test_perf, epoch)
            writer.add_scalar(f'Run{run}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Run{run}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() 
        all_test_curves.append(test_curve)
        if cfg.dataset != 'ZINC':
            wandb.run.summary["best_test" + str(run)] = maximum_test
            run_tests.append(maximum_test)
        time_average_epoch = time.time() - start_outer
        print(f'Run {run}, Vali: {best_val_perf}, Test: {test_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    all_test_curves = np.array(all_test_curves)  # (num_folds, num_epochs)
    mean_test_curve = np.mean(all_test_curves, axis=0)  # (num_epochs,)

    best_epoch = np.argmax(mean_test_curve)  
    test_at_best_epoch = all_test_curves[:, best_epoch]

    mean = np.mean(test_at_best_epoch)
    wandb.run.summary["final_test_at_best_epoch"] = float(mean)
    std = np.std(test_at_best_epoch)

    print(f"Best Epoch: {best_epoch}")
    print(f"Test Performance at Best Epoch: {mean:.4f} ± {std:.4f}")
    
    test_perf = torch.tensor(test_perfs)
    vali_perf = torch.tensor(vali_perfs)
    logger.info("-"*50)
    logger.info(config_string)
    logger.info(cfg)
    if cfg.dataset != 'ZINC':
       wandb.run.summary["mean_test"] = run_tests.mean()
    logger.info(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
    print(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

import random, numpy as np
import warnings
def set_random_seed(seed=0, cuda_deterministic=True):
    """
    This function is only used for reproducbility, 
    DDP model doesn't need to use same seed for model initialization, 
    as it will automatically send the initialized model from master node to other nodes. 
    Notice this requires no change of model after call DDP(model)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training with CUDNN deterministic setting,'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        warnings.warn('You have chosen to seed training WITHOUT CUDNN deterministic. '
                       'This is much faster but less reproducible')


import os
import shutil
from ruamel.yaml import YAML
import torch
import numpy as np
from utils.init_utils import get_trainer, get_dataloader

def init_train(cfgs, args):
    # Set rng seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Create directories
    results_dir = os.path.join('./results', cfgs['dataset'], cfgs['train_id'])
    if cfgs['load_checkpoint'] == 'None':
        if os.path.isdir(results_dir):
            raise ValueError(f'{results_dir} Already existed!')

        os.makedirs(os.path.join(results_dir, 'ckpts'))
        
        yaml = YAML()
        with open(os.path.join(results_dir, f'config_{cfgs['train_id']}.yaml'), 'w') as f:
            yaml.dump(cfgs, f)
        shutil.copy('./configs/config.yaml', results_dir)


    loaders = get_dataloader(cfgs, args)
    trainer = get_trainer(cfgs, args)

    return trainer, loaders, results_dir

def init_trainer(cfgs, args):
    return get_trainer(cfgs, args)


def init_test(cfgs, args):
    """
    Implement this if you want to test on another dataset
    """
    pass
















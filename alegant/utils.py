import os
import yaml
import torch
import random
import argparse
import numpy as np
from loguru import logger
from attrdict import AttrDict

__all__ = [
    "parse_args",
    "seed_everything",
]

def parse_args():
    """
    Parse command-line arguments and load the configuration file.

    Returns:
        config (AttrDict): Configuration loaded from the file and command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='config.yaml',
                        help='Path to the configuration file')
    # parser.add_argument('--do_test', action='store_true', default=False,
    #                 help='Train or Test, True for test, False for train')
    
    args = parser.parse_args()
    config = AttrDict(
        yaml.load(open(args.config_file, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    # Update the configuration with command-line arguments
    for k, v in vars(args).items():
        setattr(config, k, v)

    return config


def seed_everything(seed: int) -> int:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state
    """
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_log_dir(base_dir, version=None):
    if version is None:
        # Automatically find the next available version
        existing_versions = [
            int(d.split('_')[-1])
            for d in os.listdir(base_dir)
            if d.startswith("version") and d.split('_')[-1].isdigit()
        ]
        version = max(existing_versions, default=0) + 1

    return os.path.join(base_dir, f"version_{version}")
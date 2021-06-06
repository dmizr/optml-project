import logging
from typing import Union

import random
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.stats import truncnorm

def set_global_seed(seed: int, use_cuda: bool = True) -> None:

    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")


def flatten(d: Union[dict, list], parent_key: str = "", sep: str = ".") -> dict:
    """Flattens a dictionary or list into a flat dictionary

    Args:
        d: dictionary or list to flatten
        parent_key: key of parent dictionary
        sep: separator between key and child key

    Returns:
        flattened dictionary

    """
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, elem in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(elem, new_key, sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)


def truncated_normal(size, threshold=1):
    """Samples values from truncated normal distribution centered at 0

    Args:
        size: shape or amount of samples
        threshold: cut-off value for distribution

    Returns:
        numpy array of given size

    """
    return truncnorm.rvs(-threshold, threshold, size=size)


def weight_diff_norm(model: nn.Module, ema_model: nn.Module) -> float:
    """Computes the L2 norm of the difference in weights between two models"""
    l2_norm = 0

    for param1, param2 in zip(model.parameters(), ema_model.parameters()):
        l2_norm += torch.linalg.norm(param1 - param2) ** 2

    l2_norm = torch.sqrt(l2_norm).item()

    return l2_norm

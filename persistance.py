import os
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

import src.utils as utils


def visualize_persistance(preds_a, preds_b, iters: List[int] = None, k: int = 10):
    """Visualize the prediction discrepancy between two models"""

    assert preds_a.shape == preds_b.shape, "Shape of predictions must be the same"
    if iters is None:
        iters = np.arange(len(preds_a))

    match = np.mean((preds_a == preds_b), dim=0)
    top_mismatch = np.argsort(match)[:k]

    for i in range(k):
        plt.plot(np.repeat(k - 0.1, preds_a.shape[0]), preds_a[:, top_mismatch[i]])
        plt.plot(np.repeat(k + 0.1, preds_b.shape[0]), preds_b[:, top_mismatch[i]])


@hydra.main(config_path="conf", config_name="persistance")
def persistance(cfg: DictConfig):
    utils.display_config(cfg)

    path = cfg.preds_path
    preds_a, preds_b = [], []
    epoch = 0
    while os.path.isfile(os.path.join(path, f"{epoch}.preds")):
        preds_a.append(torch.load(os.path.join(path, f"{epoch}.preds")))
        preds_b.append(torch.load(os.path.join(path, f"{epoch}_average.preds")))
        epoch += 1

    preds_a = np.concatenate([p.numpy() for p in preds_a], axis=0)
    preds_b = np.concatenate([p.numpy() for p in preds_b], axis=0)
    visualize_persistance(preds_a, preds_b, [i for i in range(epoch)])


if __name__ == "__main__":
    persistance()

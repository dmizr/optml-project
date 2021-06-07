import os
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

import src.utils as utils


def visualize_persistance(preds_a, preds_b, iters: List[int] = None, k: int = 20):
    """Visualize the prediction discrepancy between two models"""

    assert preds_a.shape == preds_b.shape, "Shape of predictions must be the same"
    if iters is None:
        iters = np.arange(len(preds_a))

    match = np.mean((preds_a == preds_b), axis=0)
    top_mismatch = np.argsort(match)[:k]

    image = np.zeros((k * 3, preds_a.shape[0]))
    for i in range(k):
        image[i*3, :] = preds_a[:, top_mismatch[i]]
        image[i*3+1, :] = preds_b[:, top_mismatch[i]]
        image[i*3+2, :] = -1

    plt.imshow(image)
    plt.colorbar()
    plt.yticks([i*3 for i in range(k)], range(k))
    plt.xlabel('Epoch')
    plt.ylabel('Samples')
    plt.savefig('persistance.png')


@hydra.main(config_path="conf", config_name="persistance")
def persistance(cfg: DictConfig):
    utils.display_config(cfg)

    path = hydra.utils.to_absolute_path(cfg.preds_path)
    preds_a, preds_b = [], []
    epoch = 0

    while os.path.isfile(os.path.join(path, f"{epoch}.npy")):
        preds_a.append(np.load(os.path.join(path, f"{epoch}.npy")))
        preds_b.append(np.load(os.path.join(path, f"{epoch}_average.npy")))
        epoch += 1

    preds_a = np.stack(preds_a, axis=0)
    preds_b = np.stack(preds_b, axis=0)
    visualize_persistance(preds_a, preds_b, [i for i in range(epoch)])


if __name__ == "__main__":
    persistance()

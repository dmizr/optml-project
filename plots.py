import os
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

import src.utils as utils
from src.plots import (
    plot_misclassification,
    plot_mismatch,
    plot_persistance,
    plot_stability,
)


@hydra.main(config_path="conf", config_name="plots")
def plots(cfg: DictConfig):
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

    assert preds_a.shape == preds_b.shape, "Shape of predictions must be the same"

    labels = ["Single", "Averaged"]
    iters = np.arange(len(preds_a))

    plot_stability([preds_a, preds_b], labels, iters)
    plot_mismatch([preds_a, preds_b], labels, iters)
    plot_misclassification([preds_a, preds_b], labels, iters)
    plot_persistance([preds_a, preds_b], labels, iters, sort="stability")


if __name__ == "__main__":
    plots()

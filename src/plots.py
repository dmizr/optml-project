from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_stability(
    preds: List[np.ndarray],
    labels: Optional[List[str]] = None,
    iters: Optional[List[int]] = None,
    plot_accuracy: bool = True,
):
    """Visualize the stability of predictions"""

    sns.set_theme(context="paper", style="darkgrid")

    for pred in preds:
        assert preds[0].shape == pred.shape, "Shape of predictions must be the same"

    if iters is None:
        iters = np.arange(len(preds_a))
    if labels is None:
        labels = [str(i) for i in range(1, 1 + len(preds))]

    # prediction stability
    plt.figure()

    for i, pred in enumerate(preds):
        match = np.mean(pred[1:, ...] == pred[:-1, ...], axis=1)
        accuracy = np.mean(pred[1:, ...], axis=1)

        plt.plot(match, label=f"Stability {labels[i]}")
        plt.plot(accuracy, label=f"Accuracy {labels[i]}", linestyle="dashed")

    plt.legend()
    plt.ylabel("Stable (%)")
    plt.xlabel("Iterations")
    plt.savefig("stability.png")


def plot_mismatch(
    preds: List[np.ndarray],
    labels: Optional[List[str]] = None,
    iters: Optional[List[int]] = None,
    histogram: bool = True,
    window: int = 0,
):
    """Visualize the mismatch between predictions"""

    sns.set_theme(context="paper", style="darkgrid")

    if histogram:
        assert (
            len(preds) == 2
        ), "Only two list of predictions is supported for histogram"
    else:
        if iters is None:
            iters = np.arange(len(preds_a))
        if labels is None:
            labels = [str(i) for i in range(1, 1 + len(preds))]

    plt.figure()

    if histogram:
        match = np.mean((preds[0] == preds[1]), axis=0)
        plt.hist(1 - match, bins=100)
        plt.xlabel("Mismatch ratio (cumulative over iterations)")
        plt.ylabel("# of samples")
    else:
        for i, pred_a in enumerate(preds):
            for j, pred_b in enumerate(preds[i + 1 :]):
                match = np.mean((preds[0] == preds[1]), axis=1)
                if window > 0:
                    y = np.convolve(
                        1 - match, np.ones((window,)) / window, mode="valid"
                    )
                    x = np.linspace(window / 2, len(match) - window / 2, len(y))
                    x = np.asarray(iters)[x.astype("int")]
                else:
                    y = 1 - match
                    x = np.asarray(iters)

                plt.plot(x, y, label=f"{labels[i]} - {labels[i+j+1]}")

        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Mismatch ratio")

    plt.savefig("mismatch.png")


def plot_misclassification(
    preds: List[np.ndarray],
    labels: Optional[List[str]] = None,
    iters: Optional[List[int]] = None,
    histogram: bool = False,
    window: int = 0,
):
    """Visualize the misclassifications of predictions"""

    sns.set_theme(context="paper", style="darkgrid")

    if histogram:
        assert (
            len(preds) == 2
        ), "Only two list of predictions is supported for histogram"
    else:
        if iters is None:
            iters = np.arange(len(preds_a))
        if labels is None:
            labels = [str(i) for i in range(1, 1 + len(preds))]

    plt.figure()

    if histogram:
        accuracy_a = np.mean(preds[0][1:, ...], axis=0)
        accuracy_b = np.mean(preds[1][1:, ...], axis=0)

        plt.hist(1 - accuracy_a, bins=100, label=labels[0], alpha=0.5)
        plt.hist(1 - accuracy_b, bins=100, label=labels[1], alpha=0.5)
        plt.xlabel("Misclassification ratio (cumulative over iterations)")
        plt.ylabel("# of samples")
    else:
        for i, pred_a in enumerate(preds):
            misclassification = 1 - np.mean(pred_a, axis=1)
            if window > 0:
                y = np.convolve(
                    misclassification, np.ones((window,)) / window, mode="valid"
                )
                x = np.linspace(window / 2, len(match) - window / 2, len(y))
                x = np.asarray(iters)[x.astype("int")]
            else:
                y = misclassification
                x = np.asarray(iters)
                plt.plot(x, y, label=labels[i])

            for j, pred_b in enumerate(preds[i + 1 :]):
                misclassification = np.mean((pred_a == pred_b) * (1 - pred_a), axis=1)
                if window > 0:
                    y = np.convolve(
                        misclassification, np.ones((window,)) / window, mode="valid"
                    )
                    x = np.linspace(window / 2, len(match) - window / 2, len(y))
                    x = np.asarray(iters)[x.astype("int")]
                else:
                    y = misclassification
                    x = np.asarray(iters)

                plt.plot(x, y, label=f"{labels[i]} and {labels[i+j+1]}")

        plt.xlabel("Iterations")
        plt.ylabel("Misclassification ratio")

    plt.legend()
    plt.savefig("misclassification.png")


def plot_persistance(
    preds: List[np.ndarray],
    labels: Optional[List[str]] = None,
    iters: Optional[List[int]] = None,
    n_samples: int = 10,
    sort: str = "mismatch",
):
    """Visualize the persistance of predictions"""

    assert len(preds) == 2, "Only two list of predictions is supported for persistance"

    sns.set_theme(context="paper", style="dark")

    preds_a, preds_b = preds[0], preds[1]

    if sort == "mismatch":
        match = np.mean(preds_a == preds_b, axis=0)
        top_n = np.argsort(match)[:n_samples]
    elif sort == "misclassification":
        misclassification = np.mean(preds_a + preds_b, axis=0)
        top_n = np.argsort(misclassification)[:n_samples]
    elif sort == "stability":
        stability = np.mean(preds_a[1:, ...] == preds_a[:-1, ...], axis=0) + np.mean(
            preds_b[1:, ...] == preds_b[:-1, ...], axis=0
        )
        top_n = np.argsort(stability)[:n_samples]
    else:
        raise ValueError

    image = np.zeros((n_samples * 3, preds_a.shape[0]))
    for i in range(n_samples):
        image[i * 3, :] = preds_a[:, top_n[i]]
        image[i * 3 + 1, :] = preds_b[:, top_n[i]]
        image[i * 3 + 2, :] = -1

    plt.figure(figsize=(n_samples, 10))
    map = sns.heatmap(image, annot=False)
    plt.yticks([i * 3 for i in range(n_samples)], range(n_samples))
    plt.xlabel("Epoch")
    plt.ylabel("Samples")
    plt.savefig("persistance.png")

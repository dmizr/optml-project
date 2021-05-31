import collections
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split

def split_dataset(dataset: Dataset, split: float, seed: int) -> Tuple[Subset, Subset]:
    """Splits dataset into a train / val set based on a split value and seed

    Args:
        dataset: dataset to split
        split: The proportion of the dataset to include in the validation split,
            must be between 0 and 1.
        seed: Seed used to generate the split

    Returns:
        Subsets of the input dataset

    """
    # Verify that the dataset is Sized
    if not isinstance(dataset, collections.abc.Sized):
        raise ValueError("Dataset is not Sized!")

    if not (0 <= split <= 1):
        raise ValueError(f"Split value must be between 0 and 1. Value: {split}")

    val_length = int(len(dataset) * split)
    train_length = len(dataset) - val_length
    splits = random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits

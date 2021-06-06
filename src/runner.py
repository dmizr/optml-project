import logging
import os
from typing import Optional, Tuple

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from src.dataset import split_dataset
from src.evaluator import Evaluator
from src.scheduler import ExponentialDecayLR
from src.trainer import Trainer
from src.transform import cifar10_transform


def train(cfg: DictConfig):
    """Trains model from config

    Args:
        cfg: Hydra config

    Returns:
        Tuple of train, validation and test accuracy (on a 0 to 1 scale)

    """
    # Logger
    logger = logging.getLogger()

    # Device
    device = get_device(cfg)

    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Model
    # Use Hydra's instantiation to initialize directly from the config file

    model: torch.nn.Module = instantiate(cfg.model).to(device)
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss().to(device)
    optimizer: torch.optim.Optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)
    update_sched_on_iter = True if isinstance(scheduler, ExponentialDecayLR) else False

    # Averaged model
    averaged_model: Optional[ModelEmaV2] = (
        instantiate(cfg.averaged, model, device=device)
        if cfg.averaged is not None
        else None
    )

    # Paths
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.checkpoint)
        if cfg.checkpoint is not None
        else None
    )

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    # Trainer init
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=cfg.hparams.epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        update_sched_on_iter=update_sched_on_iter,
        grad_clip_max_norm=cfg.hparams.grad_clip_max_norm,
        writer=writer,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        averaged_model=averaged_model,
    )

    # Launch training process
    trainer.train()

    def write_eval(writer, train_acc, val_acc, test_acc, prefix=""):
        if train_acc:
            writer.add_scalar(f"Eval/Accuracy/{prefix}train", train_acc, -1)
        if val_acc:
            writer.add_scalar(f"Eval/Accuracy/{prefix}val", val_acc, -1)
        if test_acc:
            writer.add_scalar(f"Eval/Accuracy/{prefix}test", test_acc, -1)

    # Evaluate
    cfg.dataset.download = False
    accuracies = {}

    # Final model
    logger.info("Final model")
    model_path = os.path.join(save_path, "final_model.pt")
    train_acc, val_acc, test_acc = evaluate(cfg, model, model_path)
    accuracies["final"] = (train_acc, val_acc, test_acc)
    write_eval(writer, train_acc, val_acc, test_acc, prefix="final_")

    # Final averaged model
    if averaged_model is not None:
        logger.info("Final averaged model")
        model_path = os.path.join(save_path, "final_averaged_model.pt")
        train_acc, val_acc, test_acc = evaluate(cfg, model, model_path)
        accuracies["final_averaged"] = (train_acc, val_acc, test_acc)
        write_eval(writer, train_acc, val_acc, test_acc, prefix="final_averaged_")

    if val_loader is not None:
        # Best model
        logger.info("Best model")
        model_path = os.path.join(save_path, "best_model.pt")
        train_acc, val_acc, test_acc = evaluate(cfg, model, model_path)
        accuracies["best"] = (train_acc, val_acc, test_acc)
        write_eval(writer, train_acc, val_acc, test_acc, prefix="best_")

        # Best averaged model
        if averaged_model is not None:
            logger.info("Best averaged model")
            model_path = os.path.join(save_path, "best_averaged_model.pt")
            train_acc, val_acc, test_acc = evaluate(cfg, model, model_path)
            accuracies["best_averaged"] = (train_acc, val_acc, test_acc)
            write_eval(writer, train_acc, val_acc, test_acc, prefix="best_averaged")

    return accuracies


def evaluate(
    cfg: DictConfig,
    model: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[str] = None,
):
    """Evaluates model from config

    Args:
        cfg: Hydra config
        model: Overrides model from config if provided
        checkpoint_path: Overrides checkpoint from config if provided
    """
    # Logger
    logger = logging.getLogger()

    # Device
    device = get_device(cfg)

    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Model
    if model is None:
        model: torch.nn.Module = instantiate(cfg.model).to(device)

    # Load from config if checkpoint path is not provided
    if checkpoint_path is None:
        checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)

    train_acc, val_acc, test_acc = None, None, None

    if train_loader is not None:
        logger.info("Evaluating on training data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=train_loader,
            checkpoint_path=checkpoint_path,
        )
        train_acc = evaluator.evaluate()
        # Remove checkpoint loading for other loaders
        checkpoint_path = None

    if val_loader is not None:
        logger.info("Evaluating on validation data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=test_loader,
            checkpoint_path=checkpoint_path,
        )
        val_acc = evaluator.evaluate()
        # Remove checkpoint loading for other loaders
        checkpoint_path = None

    if test_loader is not None:
        logger.info("Evaluating on test data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=test_loader,
            checkpoint_path=checkpoint_path,
        )
        test_acc = evaluator.evaluate()

    return train_acc, val_acc, test_acc


def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config

    Args:
        cfg: Hydra config

    Returns:
        device on which the model will be trained or evaluated

    """
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device


def get_loaders(
    cfg: DictConfig,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Initializes the training, validation, test data & loaders from config

    Args:
        cfg: Hydra config

    Returns:
        Tuple containing the train dataloader, validation dataloader and test dataloader
    """

    train_transform = cifar10_transform(augment=cfg.dataset.train.augment)
    test_transform = cifar10_transform(augment=False)
    dataset = CIFAR10

    root = hydra.utils.to_absolute_path(cfg.dataset.root)

    # Train
    if cfg.dataset.train.use:
        train_set = dataset(
            root,
            train=True,
            transform=train_transform,
            download=cfg.dataset.download,
        )
        if cfg.dataset.val.use and cfg.dataset.val.split is not None:
            train_set, _ = split_dataset(
                dataset=train_set,
                split=cfg.dataset.val.split,
                seed=cfg.dataset.val.seed,
            )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.hparams.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        train_loader = None

    # Validation
    if cfg.dataset.val.use:
        if cfg.dataset.val.split is not None and cfg.dataset.val.split != 0.0:
            val_set = dataset(
                root,
                train=True,
                transform=test_transform,
                download=cfg.dataset.download,
            )
            _, val_set = split_dataset(
                dataset=val_set,
                split=cfg.dataset.val.split,
                seed=cfg.dataset.val.seed,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=cfg.hparams.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
            )

        else:
            logger = logging.getLogger()
            logger.info("No validation set will be used, as no split value was given.")
            val_loader = None
    else:
        val_loader = None

    # Test
    if cfg.dataset.test.use:
        test_set = dataset(
            root,
            train=False,
            transform=test_transform,
            download=cfg.dataset.download,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.hparams.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

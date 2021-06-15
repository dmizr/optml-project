# Studying weight averaging

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380//)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This codebase is used to study exponential moving averages (EMA) in the context of model weight averaging. It provides ways to experiment with different EMA decay coefficients, learning rate schedules and models on CIFAR-10.

## Table of Contents

- [Dependencies](#dependencies)
- [How to run](#how-to-run)
  - [Examples](#examples)
- [Experiment logs](#experiment-logs)
- [Project structure](#project-structure)
- [Authors](#authors)
- [Credits](#credits)


## Dependencies
This project requires Python >= 3.8. Dependencies can be installed with:
```
pip install -r requirements.txt
```


## How to run

This project uses [Hydra](https://hydra.cc/) to configure experiments. Configurations can be overridden through config files (in `conf/`) and the command line. For more information, check out the [Hydra documentation](https://hydra.cc/docs/intro/).

With Hydra, configurations can be fully customized directly though the command line. To find out more about the configuration options, run:
- For the main training experiment:
```
python3 train.py --help
```
- For hyperparameter tuning with Optuna ([more info](https://hydra.cc/docs/plugins/optuna_sweeper)):
```
python3 tune.py --help
```
- For the plots:
```
python3 plots.py --help
```

By default, run metrics are logged to [TensorBoard](https://www.tensorflow.org/tensorboard). In addition, the saved models, training parameters and training log can be found in the run's directory, in `outputs/`.

### Examples
To reproduce the results from Figure 1 of the report, run:
```
python3 train.py --multirun model=resnet20 scheduler=multistep_he,step_effnet,constant,cosine,linear averaged.decay=0.9,0.99,0.999,0.9995,0.9997 global_seed=1111,1234,4321
```
**Warning:** This code trains a ResNet-20 on CIFAR-10 **75 times**. You can train only one of these models by selecting the desired scheduler, EMA decay coefficient and seed.

To reproduce the results from Table 2 of the report, run:
```
python3 train.py --multirun model=resnet20 averaged.decay=0.9995 scheduler=constant optimizer.lr=0.5,0.2,0.1,0.05,0.02,0.01 global_seed=1234
```

## Experiment logs

The logs of our experiments are available through [tensorboard.dev](https://tensorboard.dev/).

- Scheduler + decay rate experiments (Table 1, Fig. 1): [**Log (Run 1)**](https://tensorboard.dev/experiment/p9Hjq9kySPaeg84NtZX57Q) | [**Log (Run 2)**](https://tensorboard.dev/experiment/8NJxp4UgSXamjGrbUlQUxw) | [**Log (Run 3)**](https://tensorboard.dev/experiment/Wwrw9xKiQQGzwVp1DE2d2w)
- Learning rate experiment (Table 2): [**Log**](https://tensorboard.dev/experiment/iy10jUyCQyaW62wWU7yKpw)


## Project structure


```
├── eval.py     # Hydra evaluation script
├── plots.py    # Hydra plotting script
├── train.py    # Hydra training script
├── tune.py     # Hydra + Optuna hyperparameter tuning script
│
├── conf/    # Hydra configuration directory
│   ├── eval.yaml   # Default evaluation config
│   ├── plots.yaml  # Default plotting  config
│   ├── train.yaml  # Default training config
│   ├── tune.yaml   # Default hyperparameter tuning config
│   └── (+ all directories contain hyperparameter configs for each aspect of
│        the experiments: dataset, model, optimizer, scheduler, etc...) 
│ 
└── src/     # Implementation of the experiments
    ├── dataset.py      # Utilities for splitting the data
    ├── evaluator.py    # Evaluates a trained model
    ├── metrics.py      # Metrics to keep track of the loss and accuracy
    ├── plotter.py      # Plots comparing EMA and No EMA models
    ├── runner.py       # Launches training and evaluating based on Hydra config
    ├── scheduler.py    # Learning rate schedules
    ├── trainer.py      # Training loop and logging
    ├── transform.py    # Data transforms
    ├── utils.py        # Training utilities
    │
    └── network         # Network architectures
        ├── resnet_cifar.py     # ResNets designed for CIFAR (ResNet-20, 56, ...)
        └── resnet_imagenet.py  # ResNets originally designed for ImageNet then 
                                adapted for CIFAR (ResNet-18, 50, ...)   
```

## Authors
- Julien Adda
- David Mizrahi
- Oğuz Kaan Yüksel


## Credits
Many parts of this codebase come from other repositories, namely:

- The code structure in this project is mostly adapted from [dmizr/phuber](https://github.com/dmizr/phuber).
- The EMA model comes from [timm](https://github.com/rwightman/pytorch-image-models).
- The CIFAR-10 ResNets come from [akamaster/pytorch-resnet-cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
- The learning rate schedules with warm-up come from [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch).

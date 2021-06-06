import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils as utils


@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig):
    utils.display_config(cfg)
    accuracies = runner.train(cfg)

    return accuracies["final_averaged"][1]


if __name__ == "__main__":
    tune()

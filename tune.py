import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils as utils


@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig):
    utils.display_config(cfg)
    (_, val_acc, _), (_, avg_val_acc, _) = runner.train(cfg)

    return avg_val_acc


if __name__ == "__main__":
    tune()

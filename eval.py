import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils as utils


@hydra.main(config_path="conf", config_name="eval")
def eval(cfg: DictConfig):
    utils.display_config(cfg)
    runner.evaluate(cfg)


if __name__ == "__main__":
    eval()

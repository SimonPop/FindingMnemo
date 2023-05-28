from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    experiment_name: str
    max_epochs: int = 100
    n_trials: int = 100
    seed: int = 0
    log_folder: str


def read_config() -> TrainingConfig:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    return TrainingConfig(**yaml_config)


CONFIG = read_config()

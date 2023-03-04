import yaml
from pydantic import BaseModel
from pathlib import Path
from enum import Enum

class LossType(Enum):
    Triplet = "triplet"
    Pair = "pair"
    Mixed = "mixed"

class ModelType(Enum):
    PhoneticSiamese = "phonetic_siamese"
    NaiveBaseline = "naive_baseline"

class TrainingConfig(BaseModel):
    experiment_name: str
    max_epochs: int = 100
    n_trials: int = 100
    seed: int = 0
    log_folder: str
    best_pairs_dataset: str = "best_pairs.csv"
    worst_pairs_dataset: str = "worst_pairs.csv"
    loss_type: LossType = "pair"
    model_type: ModelType = "phonetic_siamese"

def read_config() -> TrainingConfig:
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    return TrainingConfig(**yaml_config)

CONFIG = read_config()
from src.pairing.dataset.phonetic_pair_dataset import PhoneticPairDataset
from src.pairing.dataset.phonetic_triplet_dataset import PhoneticTripletDataset
from src.pairing.model.phonetic_siamese import PhoneticSiamese
from src.pairing.training.config import CONFIG, LossType
from torch.utils.data import Dataset
from pathlib import Path

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import mlflow
import torch
import optuna

def get_dataset() -> Dataset:
    """Returns a Dataset object given loss type."""
    if CONFIG.loss_type == LossType.Pair:
        dataset = PhoneticPairDataset(
            best_pairs_path=CONFIG.best_pairs_dataset, worst_pairs_path=CONFIG.worst_pairs_dataset
        )
    elif CONFIG.loss_type == LossType.Triplet:
        dataset = PhoneticTripletDataset(
            best_pairs_path=CONFIG.best_pairs_dataset, worst_pairs_path=CONFIG.worst_pairs_dataset
        )
    else:
        raise ValueError(f'Unknown loss type given: {CONFIG.loss_type}')
    return dataset

dataset = get_dataset()
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [len(dataset) - 200, 100, 100]
)


def objective(trial):
    mlf_logger = MLFlowLogger(
        experiment_name=CONFIG.experiment_name, tracking_uri=CONFIG.log_folder
    )
    trainer = Trainer(
        max_epochs=CONFIG.max_epochs,
        logger=mlf_logger,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min")],
        accelerator="gpu", devices=1
    )
    instance = instanciate(
        {
            "dropout": trial.suggest_float("dropout", 0, 0.3),
            "lr": trial.suggest_int("lr", 1, 4),
            "weight_decay": trial.suggest_int("lr", 1, 10),
            "dim_feedforward": 2 ** trial.suggest_int("dim_feedforward", 4, 10),
            "batch_size": 2 ** trial.suggest_int("batch_size", 0, 4),
            "nhead": 2 ** trial.suggest_int("nhead", 0, 3),
            "embedding_dim": 2 ** trial.suggest_int("embedding_dim", 4, 10),
            "model": CONFIG.model_type
        }
    )

    mlflow.pytorch.autolog()

    with mlflow.start_run():
        model = fit_model(
            instance["model"],
            instance["train_dataloader"],
            instance["validation_dataloader"],
            trainer,
        )

        test_loss = test_model(model, instance["test_dataloader"], trainer)[0]["test_loss"]

        mlflow.pytorch.log_model(model, "weights")
        # mlflow.log_params(trial.params)
        # mlflow.log_metric("final_test_loss", test_loss)
        # mlflow.log_param("loss_type", CONFIG.loss_type)

    return test_loss

def instanciate(kwargs):
    train_dataloader = DataLoader(
        train_set, batch_size=kwargs["batch_size"], shuffle=True, num_workers=4
    )
    validation_dataloader = DataLoader(
        val_set, batch_size=kwargs["batch_size"], num_workers=4
    )
    test_dataloader = DataLoader(
        test_set, batch_size=kwargs["batch_size"], num_workers=4
    )
    model = PhoneticSiamese(
        embedding_dim=kwargs["embedding_dim"],
        dim_feedforward=kwargs["dim_feedforward"],
        nhead=kwargs["nhead"],
        dropout=kwargs["dropout"],
        loss_type=CONFIG.loss_type,
        batch_size=kwargs["batch_size"],
        weight_decay=kwargs["weight_decay"],
        lr=kwargs["lr"],
    )
    # if torch.cuda.is_available():
    return {
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_dataloader": test_dataloader,
        "model": model,
    }


def fit_model(model, train_dataloader, validation_dataloader, trainer):
    trainer.fit(model, train_dataloader, validation_dataloader)
    return model


def test_model(model, test_dataloader, trainer):
    return trainer.test(model, test_dataloader, verbose=False)


if __name__ == "__main__":
    mlflow.set_experiment(CONFIG.experiment_name)
    seed_everything(CONFIG.seed)
    study = optuna.create_study()
    study.optimize(objective, n_trials=CONFIG.n_trials)
    study.best_params
    # TODO use profiling.
    # TODO: add regularization

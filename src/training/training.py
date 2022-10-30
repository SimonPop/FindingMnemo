from src.dataset.phonetic_pair_dataset_v2 import PhoneticPairDataset
from src.model.sound_siamese_v2 import SoundSiamese
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
import optuna
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

seed_everything(0)

dataset = PhoneticPairDataset(
    best_pairs_path="best_pairs.csv", worst_pairs_path="worst_pairs.csv"
)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [len(dataset) - 200, 100, 100]
)


def objective(trial):
    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri="file:./mlruns"
    )
    trainer = Trainer(
        max_epochs=10,
        logger=mlf_logger,
        # callbacks=[EarlyStopping(monitor="validation_loss", mode="min")]
    )
    instance = instanciate(
        {
            "dropout": trial.suggest_float("dropout", 0, 0.3),
            "dim_feedforward": 2 ** trial.suggest_int("dim_feedforward", 4, 10),
            "batch_size": 2 ** trial.suggest_int("batch_size", 0, 4),
            "nhead": 2 ** trial.suggest_int("nhead", 0, 3),
            "embedding_dim": 2 ** trial.suggest_int("embedding_dim", 4, 10),
        }
    )
    model = fit_model(
        instance["model"],
        instance["train_dataloader"],
        instance["validation_dataloader"],
        trainer,
    )
    return test_model(model, instance["test_dataloader"], trainer)[0]["test_loss"]


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
    model = SoundSiamese(
        embedding_dim=kwargs["embedding_dim"],
        dim_feedforward=kwargs["dim_feedforward"],
        nhead=kwargs["nhead"],
        dropout=kwargs["dropout"],
    )
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
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    study.best_params

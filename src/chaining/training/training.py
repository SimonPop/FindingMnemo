import mlflow
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, Dataset

from src.chaining.dataset.graph_distance_dataset import GraphDistanceDataset
from src.chaining.model.distance_estimator import DistanceEstimator
from src.chaining.training.config import CONFIG

train_set = GraphDistanceDataset(mode="train")
val_set = GraphDistanceDataset(mode="validation")
test_set = GraphDistanceDataset(mode="test")


def objective(trial):
    mlf_logger = MLFlowLogger(
        experiment_name=CONFIG.experiment_name, tracking_uri=CONFIG.log_folder
    )
    trainer = Trainer(
        max_epochs=CONFIG.max_epochs,
        logger=mlf_logger,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min")],
    )
    instance = instanciate(
        {
            "dropout": trial.suggest_float("dropout", 0, 0.3),
            "dim_feedforward": 2 ** trial.suggest_int("dim_feedforward", 4, 10),
            "batch_size": 2 ** trial.suggest_int("batch_size", 0, 4),
            "nhead": 2 ** trial.suggest_int("nhead", 0, 3),
            "embedding_dim": 2 ** trial.suggest_int("embedding_dim", 4, 10),
            "model": CONFIG.model_type,
        }
    )
    model = fit_model(
        instance["model"],
        instance["train_dataloader"],
        instance["validation_dataloader"],
        trainer,
    )

    test_loss = test_model(model, instance["test_dataloader"], trainer)[0]["test_loss"]

    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_params(trial.params)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_param("loss_type", CONFIG.loss_type)

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
    model = DistanceEstimator(
        embedding_dim=kwargs["embedding_dim"],
        dim_feedforward=kwargs["dim_feedforward"],
        batch_size=kwargs["batch_size"],
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
    seed_everything(CONFIG.seed)
    study = optuna.create_study()
    study.optimize(objective, n_trials=CONFIG.n_trials)
    study.best_params
    # TODO use profiling.

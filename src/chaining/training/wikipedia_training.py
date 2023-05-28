import mlflow
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression

from src.chaining.dataset.wikipedia_dataset import WikipediaDataset

mlflow.set_experiment("Wikipedia")

generator = WikipediaDataset(hop_nb=200)

train_set, _ = generator.create_dataset(pair_nb=5000)
test_set, _ = generator.create_dataset(pair_nb=500)

X_train = train_set[
    [
        "common_cat",
        "degree_sum",
        "degree_diff",
        "similarity",
        "distance",
        "percent_cat",
        "tfidf",
    ]
]
y_train = train_set["length"]

X_test = test_set[
    [
        "common_cat",
        "degree_sum",
        "degree_diff",
        "similarity",
        "distance",
        "percent_cat",
        "tfidf",
    ]
]
y_test = test_set["length"]


def objective(trial):
    trial.suggest_categorical("model_type", ["linear", "elastic", "forest"])
    if trial.params["model_type"] == "forest":
        trial.suggest_int("n_estimators", 50, 100)
        trial.suggest_int("max_depth", 3, 100)
    model = instanciate_model(**trial.params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(trial.params)
        mlflow.log_metric("score", score)

    return score


def instanciate_model(model_type: str, **kwargs) -> BaseEstimator:
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "elastic":
        return ElasticNet()
    elif model_type == "forest":
        return RandomForestRegressor(
            n_estimators=kwargs["n_estimators"], max_depth=kwargs["max_depth"]
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)

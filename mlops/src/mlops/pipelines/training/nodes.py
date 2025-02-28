import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from hyperopt import hp, tpe, fmin, Trials
import warnings
import os
import mlflow
from mlflow.models.signature import infer_signature
from matplotlib import pyplot as plt

# from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")

"""    {
        "name": "CatBoost Regressor",
        "class": CatBoostRegressor,
        "params": {
            "loss_function": "RMSE",  # Équivalent à l'objectif 'regression' de LightGBM
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
            "iterations": hp.quniform(
                "n_estimators", 100, 1000, 50
            ),  # Équivalent à n_estimators
            "depth": hp.quniform("max_depth", 4, 12, 1),  # Équivalent à max_depth
            "l2_leaf_reg": hp.choice(
                "reg_lambda", [0, 1e-1, 1, 2, 5, 10]
            ),  # Équivalent à reg_lambda
            "bagging_temperature": hp.uniform(
                "subsample", 0.5, 1
            ),  # Équivalent à subsample
            "border_count": hp.quniform(
                "border_count", 8, 128, 10
            ),  # Équivalent correct
            "min_data_in_leaf": hp.quniform(
                "min_child_samples", 1, 20, 1
            ),  # Équivalent à min_child_samples
            "verbose": 0,  # Désactive les logs
        },
        "override_schemas": {
            "depth": int,
            "iterations": int,
            "border_count": int,
            "min_data_in_leaf": int,
        },
    },
"""

MODELS = [
    {
        "name": "LightGBM Regressor",
        "class": LGBMRegressor,
        "params": {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
            "n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
            "max_depth": hp.quniform("max_depth", 4, 12, 1),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 1),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "n_estimators": int,
        },
    },
    {
        "name": "LightGBM Regressor",
        "class": LGBMRegressor,
        "params": {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
            "n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
            "max_depth": hp.quniform("max_depth", 4, 12, 1),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 1),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "n_estimators": int,
        },
    },
]


def evaluate_model(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_model(
    instance: BaseEstimator,
    training_set: Tuple[np.ndarray, np.ndarray],
    params: Dict = {},
) -> BaseEstimator:
    filtered_models = next((x for x in MODELS if x["class"] == instance), None)
    if not filtered_models:
        raise ValueError(f"Modèle {instance} non trouvé dans MODELS")

    override_schemas = filtered_models["override_schemas"]
    params = {
        k: override_schemas[k](v) if k in override_schemas else v
        for k, v in params.items()
    }

    model = instance(**params)
    model.fit(*training_set)
    return model


def optimize_hyp(
    instance: BaseEstimator,
    dataset: Tuple[np.ndarray, np.ndarray],
    search_space: Dict,
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> Dict:
    X, y = dataset

    def objective(params):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmse_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            model = train_model(instance, (X_train, y_train), params)
            preds = model.predict(X_test)
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        return np.mean(rmse_scores)

    trials = Trials()
    return fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset: np.ndarray,
    max_evals: int = 40,
    log_to_mlflow: bool = False,
    experiment_id: int = -1,
) -> BaseEstimator:

    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))

    run_id = ""
    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        run = mlflow.start_run(experiment_id=experiment_id)
        run_id = run.info.run_id

    opt_models = []
    for model_specs in MODELS:
        optimum_params = optimize_hyp(
            model_specs["class"],
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda x, y: np.sqrt(mean_squared_error(x, y)),
            max_evals=max_evals,
        )
        model = train_model(model_specs["class"], (X_train, y_train), optimum_params)
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test, predictions)

        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "metrics": metrics,
            }
        )

    best_model = max(opt_models, key=lambda x: x["metrics"]["rmse"])

    if log_to_mlflow:
        signature = infer_signature(X_train, best_model["model"].predict(X_train))
        plot_residuals(y_test, best_model["model"].predict(X_test))

        mlflow.log_metrics(best_model["metrics"])
        mlflow.log_params(best_model["params"])
        X = dataset.drop("SalePrice", axis=1)
        np.savetxt(
            "data/08_reporting/predictions.csv",
            best_model["model"].predict(X),
            delimiter=",",
        )
        mlflow.log_artifact("data/08_reporting/predictions.csv")
        mlflow.log_artifacts("data/08_reporting", artifact_path="plots")
        mlflow.log_artifact("data/04_feature/transform_pipeline.pkl")
        mlflow.sklearn.log_model(best_model["model"], "model", signature=signature)
        mlflow.end_run()

    return dict(model=best_model, mlflow_run_id=run_id)


def plot_residuals(y_true, y_pred, filename="residuals.png"):
    y_true = np.ravel(y_true)  # Ensure y_true is a 1D NumPy array
    y_pred = np.ravel(y_pred)  # Ensure y_pred is a 1D NumPy array

    residuals = y_true - y_pred  # Now subtraction will work

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    # Save and log to MLflow
    filepath = os.path.join("mlflow_plots", filename)
    os.makedirs("mlflow_plots", exist_ok=True)
    plt.savefig(filepath)
    plt.close()

    mlflow.log_artifact(filepath, artifact_path="plots")

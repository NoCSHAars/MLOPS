import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from hyperopt import hp, tpe, fmin, Trials
import warnings
import os
import mlflow

warnings.filterwarnings("ignore")

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
    }
]


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
    max_evals: int = 40,
    log_to_mlflow: bool = False,
    experiment_id: int = -1,
) -> Dict:
    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))

    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        mlflow.start_run(experiment_id=experiment_id)

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
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "rmse": rmse,
            }
        )

    return {"model": min(opt_models, key=lambda x: x["rmse"])["model"]}

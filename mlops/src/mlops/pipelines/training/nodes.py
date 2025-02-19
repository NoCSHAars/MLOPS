import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from hyperopt import hp, tpe, fmin, Trials
import warnings
warnings.filterwarnings('ignore')

MODELS = [
    {"name": "LightGBM",
    "class": LGBMClassifier,
    "params": {
        "objective": "multiclass",  
        "num_class": 3,  
        "verbose": -1,
        "learning_rate": hp.uniform("learning_rate", 0.001, 1),
        "num_iterations": hp.quniform("num_iterations", 100, 1000, 20),
        "max_depth": hp.quniform("max_depth", 4, 12, 6),
        "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "min_child_samples": hp.quniform("min_child_samples", 1, 20, 10),
        "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
        "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
    },
    "override_schemas": {
        "num_leaves": int,
        "min_child_samples": int,
        "max_depth": int,
        "num_iterations": int,
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
    params = {k: override_schemas[k](v) if k in override_schemas else v for k, v in params.items()}
    
    if "objective" in params and params["objective"] == "multiclass":
        params["num_class"] = len(np.unique(training_set[1]))
    
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
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            model = train_model(instance, (X_train, y_train), params)
            scores.append(metric(y_test, model.predict(X_test)))
        return np.mean(scores)
    trials = Trials()
    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40
) -> Dict:
    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))
    opt_models = []
    for model_specs in MODELS:
        optimum_params = optimize_hyp(
            model_specs["class"],
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda x, y: -f1_score(x, y, average="weighted"),
            max_evals=max_evals
        )
        model = train_model(model_specs["class"], (X_train, y_train), optimum_params)
        opt_models.append({
            "model": model,
            "name": model_specs["name"],
            "params": optimum_params,
            "score": f1_score(y_test, model.predict(X_test), average="weighted"),
        })
    return max(opt_models, key=lambda x: x["score"])

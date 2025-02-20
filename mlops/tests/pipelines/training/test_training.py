from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from mlops.src.mlops.pipelines.training.nodes import train_model, optimize_hyp, auto_ml
from hyperopt import hp


def test_train_model(sample_data):
    X, y = sample_data
    model = train_model(
        LGBMRegressor, (X, y), params={"n_estimators": 100, "max_depth": 5}
    )

    assert isinstance(model, LGBMRegressor)
    assert hasattr(model, "predict")

    preds = model.predict(X)
    assert len(preds) == len(y)
    assert mean_squared_error(y, preds) >= 0  # RMSE cannot be negative


def test_optimize_hyp(sample_data):
    X, y = sample_data
    search_space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
        "max_depth": hp.quniform("max_depth", 4, 12, 1),
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
    }

    best_params = optimize_hyp(
        LGBMRegressor, (X, y), search_space, mean_squared_error, max_evals=2
    )

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params


def test_auto_ml(sample_data):
    X, y = sample_data
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    result = auto_ml(X_train, y_train, X_test, y_test, max_evals=2, log_to_mlflow=False)

    assert isinstance(result, dict)
    assert "model" in result
    assert "mlflow_run_id" in result
    assert isinstance(result["model"], dict)
    assert "rmse" in result["model"]

import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from mlops.src.mlops.pipelines.training.nodes import (
    train_model,
    optimize_hyp,
    auto_ml,
)


def test_train_model(dummy_data):
    """Tests that train_model returns a trained model."""
    X_train, y_train, _, _ = dummy_data
    params = {"learning_rate": 0.1, "num_leaves": 31}

    with patch.object(LGBMClassifier, "fit", return_value=None) as mock_fit:
        model = train_model(LGBMClassifier, (X_train, y_train), params)
        mock_fit.assert_called_once()
        assert isinstance(model, LGBMClassifier)


def test_optimize_hyp(dummy_data):
    """Tests optimize_hyp function."""
    X_train, y_train, _, _ = dummy_data
    search_space = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 6,
        "num_iterations": 100,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    with patch("mlops.src.mlops.pipelines.training.nodes.train_model") as mock_train:
        mock_train.return_value = MagicMock(
            predict=lambda x: np.random.randint(0, 2, len(x))
        )
        best_params = optimize_hyp(
            LGBMClassifier,
            (X_train, y_train),
            search_space,
            lambda x, y: f1_score(x, y, average="weighted"),
            max_evals=5,
        )

        assert isinstance(best_params, dict)  # Ensure it returns a dictionary of params


def test_auto_ml(dummy_data):
    """Tests auto_ml function."""
    X_train, y_train, X_test, y_test = dummy_data

    with patch(
        "mlops.src.mlops.pipelines.training.nodes.optimize_hyp",
        return_value={"learning_rate": 0.1},
    ), patch("mlops.src.mlops.pipelines.training.nodes.train_model") as mock_train:
        mock_model = LGBMClassifier()  # Crée une instance réelle de LGBMClassifier
        mock_model.fit = MagicMock(return_value=None)  # Mock la méthode fit
        mock_model.predict = MagicMock(
            return_value=np.random.randint(0, 2, len(y_test))
        )
        mock_train.return_value = mock_model

        result = auto_ml(X_train, y_train, X_test, y_test)

        assert "model" in result
        assert isinstance(result["model"], LGBMClassifier)
        assert "score" in result
        assert isinstance(result["score"], float)
        assert "name" in result
        assert isinstance(result["name"], str)
        assert "params" in result
        assert isinstance(result["params"], dict)

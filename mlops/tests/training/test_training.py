import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from mlops.src.mlops.pipelines.training.nodes import train_model, optimize_hyp, auto_ml  # Update with actual import path


@pytest.fixture
def dummy_data():
    """Generates dummy dataset for testing."""
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
    y_test = pd.Series(np.random.randint(0, 2, 20))
    return X_train, y_train, X_test, y_test


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
        "num_leaves": 31
    }

    with patch("mlops.src.mlops.pipelines.training.nodes.train_model") as mock_train:
        mock_train.return_value = MagicMock(predict=lambda x: np.random.randint(0, 2, len(x)))
        best_params = optimize_hyp(LGBMClassifier, (X_train, y_train), search_space, f1_score, max_evals=5)

        assert isinstance(best_params, dict)  # Ensure it returns a dictionary of params


def test_auto_ml(dummy_data):
    """Tests auto_ml function."""
    X_train, y_train, X_test, y_test = dummy_data

    with patch("mlops.src.mlops.pipelines.training.nodes.optimize_hyp", return_value={"learning_rate": 0.1}), \
            patch("mlops.src.mlops.pipelines.training.nodes.train_model") as mock_train:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randint(0, 2, len(y_test))
        mock_train.return_value = mock_model

        result = auto_ml(X_train, y_train, X_test, y_test)

        assert "model" in result
        assert isinstance(result["model"], dict)
        assert "score" in result["model"]

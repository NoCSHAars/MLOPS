import pytest
from mlops.src.mlops.pipelines.processing.nodes import encode_features, split_dataset
import pandas as pd

MIN_SAMPLES = 100


def test_encode_features(dataset_not_encoded):
    df = encode_features(dataset_not_encoded)["features"]
    assert df.shape[0] > MIN_SAMPLES


def test_split_dataset():

    data = {
        "LotFrontage": [65, 80, 68, 80, 90, 99999],
        "LotArea": [8450, 9600, 11250, 1000, 100, 200],
        "LotConfig_Inside": [True, True, False, False, True, False],
        "SalePrice": [100000, 150000, 200000, 250000, 300000, 350000],
    }
    df = pd.DataFrame(data)

    test_ratio = 0.33
    result = split_dataset(df, test_ratio)

    X_train, y_train, X_test, y_test = (
        result["X_train"],
        result["y_train"],
        result["X_test"],
        result["y_test"],
    )

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

    assert "SalePrice" not in X_train.columns
    assert "SalePrice" not in X_test.columns

    assert y_train.name == "SalePrice"
    assert y_test.name == "SalePrice"

    assert len(X_test) == pytest.approx(len(df) * test_ratio, rel=0.1)
    assert len(X_train) == pytest.approx(len(df) * (1 - test_ratio), rel=0.1)

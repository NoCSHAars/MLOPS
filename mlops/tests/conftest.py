import pytest
import numpy as np
import pandas as pd
from mlops.src.mlops.pipelines.loading.nodes import load_csv_from_bucket
from mlops.src.mlops.pipelines.processing.nodes import encode_features


@pytest.fixture(scope="module")
def project_id():
    return "enhanced-kiln-451309-q8"


@pytest.fixture(scope="module")
def primary_folder():
    return "housing-price-predict/primary"


@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, primary_folder):
    return load_csv_from_bucket(project_id, primary_folder)


@pytest.fixture(scope="module")
def test_ratio():
    return 0.3


@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    return encode_features(dataset_not_encoded)["features"]


@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.rand(100) * 100000, name="target")
    return X, y

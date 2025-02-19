import pytest

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

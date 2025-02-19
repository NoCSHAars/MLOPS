import pytest

from mlops.src.mlops.pipelines.loading.nodes import load_csv_from_bucket


@pytest.fixture(scope="module")
def project_id():
    return "enhanced-kiln-451309-q8"


@pytest.fixture(scope="module")
def primary_folder():
    return "housing-price-predict/primary"


@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, primary_folder):
    return load_csv_from_bucket(project_id, primary_folder)

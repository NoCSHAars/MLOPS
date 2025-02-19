import pytest

@pytest.fixture(scope="module")
def project_id():
    return "enhanced-kiln-451309-q8"

@pytest.fixture(scope="module")
def primary_folder():
    return "housing-price-predict/primary/"

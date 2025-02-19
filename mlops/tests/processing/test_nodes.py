from mlops.src.mlops.pipelines.processing.nodes import encode_features

MIN_SAMPLES = 500


def test_encode_features(dataset_not_encoded):
    df = encode_features(dataset_not_encoded)["features"]
    # Checking that we have enough samples
    assert df.shape[0] > MIN_SAMPLES
    print(df.head())

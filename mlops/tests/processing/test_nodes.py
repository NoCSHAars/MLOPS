from mlops.src.mlops.pipelines.processing.nodes import encode_features


def test_encode_features(dataset_not_encoded):
    df = encode_features(dataset_not_encoded)["features"]
    print(df.head())

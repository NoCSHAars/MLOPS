import pandas as pd

from typing import Dict, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features of data file.
    """
    features = dataset.drop(["ID"], axis=1).copy()

    encoders = []
    for label in ["category", "sub_category", "brand"]:
        features[label] = features[label].astype(str)
        features.loc[features[label] == "nan", label] = "unknown"
        encoder = LabelEncoder()
        features.loc[:, label] = encoder.fit_transform(features.loc[:, label].copy())
        encoders.append((label, encoder))

    features["weekday"] = features["weekday"].astype(int)
    return dict(features=features, transform_pipeline=encoders)
    
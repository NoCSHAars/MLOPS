import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_features(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Encode les colonnes catégoriques du dataset.
    """
    # Exclure la colonne Id
    features = dataset.drop(columns=["Id"], errors='ignore').copy()
    
    # Détection automatique des colonnes catégoriques (non numériques)
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    for col in categorical_columns:
        
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col])
        encoders[col] = encoder
    
    return dict(features=features, transform_pipeline=encoders)

def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Splits dataset into a training set and a test set.
    """
    X = dataset.drop("SalePrice", axis=1)
    y = dataset["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=40
    )

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)   
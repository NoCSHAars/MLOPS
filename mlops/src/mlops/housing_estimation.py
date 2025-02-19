import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import mlflow

X_train = pd.read_csv("mlops/data/05_model_input/X_train.csv")
X_test = pd.read_csv("mlops/data/05_model_input/X_test.csv")
y_train = pd.read_csv("mlops/data/05_model_input/y_train.csv")
y_test = pd.read_csv("mlops/data/05_model_input/y_test.csv")

# Hyper-paramètres des modèles
hyp_params = {
    "num_leaves": 60,
    "min_child_samples": 10,
    "max_depth": 12,
    "n_estimators": 100,
    "learning_rate": 0.1,
}

# Identification de l'interface MLflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("housing_estimation")

with mlflow.start_run() as run:
    model = LGBMRegressor(**hyp_params, objective="regression")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_params(hyp_params)
    mlflow.log_metric("rmse", rmse)

    print(mlflow.get_artifact_uri())

    # Inférer la signature du modèle
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[0:1].copy()

    mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=input_example
    )

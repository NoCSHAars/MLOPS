import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

matplotlib.use("Agg")

X_train = pd.read_csv("data/05_model_input/X_train.csv")
X_test = pd.read_csv("data/05_model_input/X_test.csv")
y_train = pd.read_csv("data/05_model_input/y_train.csv")
y_test = pd.read_csv("data/05_model_input/y_test.csv")

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


client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
client.get_metric_history(run.info.run_id, key="rmse")


def plot_residuals(y_true, y_pred, filename="residuals.png"):
    y_true = np.ravel(y_true)  # Ensure y_true is a 1D NumPy array
    y_pred = np.ravel(y_pred)  # Ensure y_pred is a 1D NumPy array

    residuals = y_true - y_pred  # Now subtraction will work

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    # Save and log to MLflow
    filepath = os.path.join("mlflow_plots", filename)
    os.makedirs("mlflow_plots", exist_ok=True)
    plt.savefig(filepath)
    plt.close()

    mlflow.log_artifact(filepath, artifact_path="plots")


def train_model(params):
    with mlflow.start_run():
        model = LGBMRegressor(**params, objective="regression")
        model.fit(X_train, y_train)

        # Predict and calculate RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log hyperparameters and RMSE
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        plot_residuals(y_test, y_pred)

        # Inferring the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_test.iloc[0:1].copy()

        # Log the model in MLflow
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )

        print(f"Logged RMSE: {rmse:.4f}")


train_model({**hyp_params, **{"n_estimators": 200, "learning_rate": 0.05}})
train_model({**hyp_params, **{"n_estimators": 500, "learning_rate": 0.025}})
train_model({**hyp_params, **{"n_estimators": 1000, "learning_rate": 0.01}})

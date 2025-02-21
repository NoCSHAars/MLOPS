from kedro.pipeline import Pipeline
from mlops.src.mlops.pipelines.training.pipeline import create_pipeline
from mlops.src.mlops.pipelines.training.nodes import auto_ml


def test_create_pipeline():
    """Test that the pipeline is created correctly."""
    pipeline = create_pipeline()

    # Ensure pipeline is an instance of Kedro Pipeline
    assert isinstance(pipeline, Pipeline)

    # Ensure pipeline has exactly one node
    assert len(pipeline.nodes) == 1

    # Get the first node
    test_node = pipeline.nodes[0]

    # Ensure the node is calling the correct function
    assert test_node.func == auto_ml

    # Ensure the inputs and outputs are correct
    assert test_node.inputs == [
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "dataset",
        "params:automl_max_evals",
        "params:mlflow_enabled",  # Added missing input
        "params:mlflow_experiment_id",  # Added missing input
    ]

    assert test_node.outputs == [
        "model",
        "mlflow_run_id",
    ]  # Updated output to match pipeline

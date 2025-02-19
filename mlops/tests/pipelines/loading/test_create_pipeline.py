from kedro.pipeline import Pipeline
from mlops.src.mlops.pipelines.loading.pipeline import create_pipeline
from mlops.src.mlops.pipelines.loading.nodes import load_csv_from_bucket


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
    assert test_node.func == load_csv_from_bucket

    # Ensure the inputs and outputs are correct
    assert test_node.inputs == ["params:gcp_project_id", "params:gcs_primary_folder"]
    assert test_node.outputs == ["primary"]

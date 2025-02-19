from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

# Assuming the pipeline is defined in a module named `pipeline.py`
from mlops.src.mlops.pipelines.processing.pipeline import create_pipeline
from mlops.src.mlops.pipelines.processing.nodes import encode_features, split_dataset


def test_create_pipeline():
    # Create the pipeline
    pipeline = create_pipeline()

    # Check if the pipeline is an instance of Pipeline
    assert isinstance(pipeline, Pipeline)

    # Get the list of nodes in the pipeline
    nodes = pipeline.nodes

    # Check the number of nodes in the pipeline
    assert len(nodes) == 2

    # Test the first node (encode_features)
    _test_node(
        nodes[0],
        expected_func=encode_features,
        expected_inputs=["primary"],
        expected_outputs=["dataset", "transform_pipeline"],
    )

    # Test the second node (split_dataset)
    _test_node(
        nodes[1],
        expected_func=split_dataset,
        expected_inputs=["dataset", "params:test_ratio"],
        expected_outputs=["X_train", "y_train", "X_test", "y_test"],
    )


def _test_node(node: Node, expected_func, expected_inputs, expected_outputs):
    """Helper function to test a node's properties."""
    # Check if the node is an instance of Node
    assert isinstance(node, Node)

    # Check if the node's function matches the expected function
    assert node.func == expected_func

    # Check if the node's inputs match the expected inputs
    assert node.inputs == expected_inputs

    # Check if the node's outputs match the expected outputs
    assert node.outputs == expected_outputs

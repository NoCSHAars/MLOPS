"""Project pipelines."""

from kedro.pipeline import Pipeline
from mlops.pipelines.training import pipeline as training_pipeline
from mlops.pipelines.processing import pipeline as processing_pipeline
from mlops.pipelines.loading import pipeline as loading_pipeline
from mlops.pipelines.deployment import pipeline as deployment_pipeline
from typing import Dict


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    p_processing = processing_pipeline.create_pipeline()
    p_training = training_pipeline.create_pipeline()
    p_loading = loading_pipeline.create_pipeline()
    p_deployment = deployment_pipeline.create_pipeline()

    return {
        "__default__": Pipeline([p_loading, p_processing, p_training, p_deployment]),
        "global": Pipeline([p_loading, p_processing, p_training, p_deployment]),
        "loading": p_loading,
        "processing": p_processing,
        "training": p_training,
        "deployment": p_deployment,
    }

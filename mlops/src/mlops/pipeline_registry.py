"""Project pipelines."""

from kedro.pipeline import Pipeline
from mlops.pipelines.training import pipeline as training_pipeline
from mlops.pipelines.processing import pipeline as processing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    p_processing = processing_pipeline.create_pipeline()
    p_training = training_pipeline.create_pipeline()

    return {
        "processing": p_processing,
        "training": p_training
    }

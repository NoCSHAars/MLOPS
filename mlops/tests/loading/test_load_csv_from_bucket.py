import pandas as pd

from mlops.src.mlops.pipelines.loading.nodes import load_csv_from_bucket

def test_load_csv_from_bucket(project_id, primary_folder):
    df = load_csv_from_bucket(project_id, primary_folder)
    print(df.head())
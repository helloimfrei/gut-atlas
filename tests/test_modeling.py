from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from gutatlas.dataset import TARGET_COLUMN
from gutatlas.modeling import evaluate_saved_models, train_models


def test_train_save_and_evaluate_logistic_artifact(tmp_path: Path) -> None:
    values, labels = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    dataset = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(values, columns=[f"feature_{index}" for index in range(6)])
    frame[TARGET_COLUMN] = labels
    frame.to_parquet(dataset, index=False)

    run_dir = tmp_path / "run"
    results = train_models(
        dataset,
        run_dir,
        ["logistic"],
        cv_splits=2,
        n_iter=1,
        n_jobs=1,
    )

    artifact_path = results[0].artifact_path
    assert artifact_path.is_file()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "features" / "logistic_coefficients.csv").is_file()

    evaluation_dir = tmp_path / "evaluation"
    evaluated = evaluate_saved_models(dataset, [artifact_path], evaluation_dir)

    assert evaluated[0][1].roc_auc == pytest.approx(results[0].metrics.roc_auc)
    assert (evaluation_dir / "metrics.json").is_file()

"""Reproducible model training and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from gutatlas.dataset import load_training_dataset
from gutatlas.features import short_taxon_name

ModelName = Literal["logistic", "xgboost", "lightgbm"]
SUPPORTED_MODELS: tuple[ModelName, ...] = ("logistic", "xgboost", "lightgbm")
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class BinaryMetrics:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ModelArtifact:
    artifact_version: int
    model_name: ModelName
    estimator: Any
    feature_names: tuple[str, ...]
    test_indices: tuple[int, ...]
    target_column: str
    random_state: int
    test_size: float
    cv_roc_auc: float
    best_params: dict[str, object]


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    metrics: BinaryMetrics
    artifact_path: Path


def _make_estimator(
    model_name: ModelName, random_state: int
) -> tuple[Any, dict[str, Any]]:
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(
            l1_ratio=0.5,
            max_iter=5_000,
            random_state=random_state,
            solver="saga",
        )
        search_space = {
            "C": Real(1e-3, 10.0, prior="log-uniform"),
            "l1_ratio": Real(0.0, 1.0),
        }
        return estimator, search_space

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError, ValueError) as error:
            raise RuntimeError(
                "XGBoost could not load. On macOS, install its OpenMP runtime "
                "with `brew install libomp`."
            ) from error

        estimator = XGBClassifier(
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
            tree_method="hist",
        )
        search_space = {
            "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
            "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
            "n_estimators": Integer(50, 800),
            "max_depth": Integer(3, 8),
        }
        return estimator, search_space

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "LightGBM could not load. On macOS, install its OpenMP runtime "
                "with `brew install libomp`."
            ) from error

        estimator = LGBMClassifier(
            n_jobs=1,
            random_state=random_state,
            verbosity=-1,
        )
        search_space = {
            "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
            "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
            "n_estimators": Integer(50, 800),
            "max_depth": Integer(3, 8),
            "num_leaves": Integer(20, 150),
        }
        return estimator, search_space

    raise ValueError(f"Unsupported model: {model_name}")


def _as_builtin(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def split_indices(
    target: pd.Series, *, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    indices = np.arange(len(target))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return np.asarray(train_indices), np.asarray(test_indices)


def predict_probabilities(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator does not provide predict_proba")
    probabilities = np.asarray(estimator.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary class probabilities")
    return probabilities[:, 1]


def calculate_metrics(
    target: pd.Series | np.ndarray, probabilities: np.ndarray
) -> BinaryMetrics:
    predicted = (probabilities >= 0.5).astype("int8")
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predicted)),
        precision=precision,
        recall=recall,
        specificity=specificity,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def train_model(
    model_name: ModelName,
    features: pd.DataFrame,
    target: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    cv_splits: int,
    n_iter: int,
    n_jobs: int,
    random_state: int,
    test_size: float,
) -> tuple[ModelArtifact, BinaryMetrics]:
    """Tune one model and evaluate it on the shared held-out split."""

    if cv_splits < 2:
        raise ValueError("cv_splits must be at least two")
    if n_iter < 1:
        raise ValueError("n_iter must be positive")

    train_target = target.iloc[train_indices]
    if int(train_target.value_counts().min()) < cv_splits:
        raise ValueError("Each training class must contain at least cv_splits rows")

    estimator, search_space = _make_estimator(model_name, random_state)
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        error_score="raise",
    )
    search.fit(features.iloc[train_indices], train_target)
    best_estimator = search.best_estimator_
    probabilities = predict_probabilities(best_estimator, features.iloc[test_indices])
    metrics = calculate_metrics(target.iloc[test_indices], probabilities)
    best_params = {
        str(key): _as_builtin(value) for key, value in search.best_params_.items()
    }
    artifact = ModelArtifact(
        artifact_version=ARTIFACT_VERSION,
        model_name=model_name,
        estimator=best_estimator,
        feature_names=tuple(map(str, features.columns)),
        test_indices=tuple(map(int, test_indices)),
        target_column=str(target.name)
        if target.name is not None
        else "disease_present",
        random_state=random_state,
        test_size=test_size,
        cv_roc_auc=float(search.best_score_),
        best_params=best_params,
    )
    return artifact, metrics


def save_artifact(artifact: ModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> ModelArtifact:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(f"Not a Gut Atlas model artifact: {path}")
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {artifact.artifact_version}: {path}"
        )
    return artifact


def evaluate_artifact(
    artifact: ModelArtifact, features: pd.DataFrame, target: pd.Series
) -> tuple[BinaryMetrics, np.ndarray]:
    expected = list(artifact.feature_names)
    if list(map(str, features.columns)) != expected:
        raise ValueError("Dataset features do not match the model artifact")
    if not artifact.test_indices or max(artifact.test_indices) >= len(features):
        raise ValueError("Artifact test indices do not match the dataset")

    test_indices = list(artifact.test_indices)
    probabilities = predict_probabilities(
        artifact.estimator, features.iloc[test_indices]
    )
    return calculate_metrics(target.iloc[test_indices], probabilities), probabilities


def _write_coefficient_outputs(artifact: ModelArtifact, output_dir: Path) -> None:
    if artifact.model_name != "logistic" or not hasattr(artifact.estimator, "coef_"):
        return
    coefficients = np.asarray(artifact.estimator.coef_)[0]
    frame = pd.DataFrame(
        {
            "feature": artifact.feature_names,
            "taxon": [short_taxon_name(name) for name in artifact.feature_names],
            "coefficient": coefficients,
        }
    )
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    frame = frame.sort_values("absolute_coefficient", ascending=False)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(feature_dir / "logistic_coefficients.csv", index=False)
    frame[frame["coefficient"] < 0].head(20).to_csv(
        feature_dir / "logistic_protective.csv", index=False
    )
    frame[frame["coefficient"] > 0].head(20).to_csv(
        feature_dir / "logistic_risk_enhancing.csv", index=False
    )


def _write_run_outputs(
    results: list[TrainingResult],
    output_dir: Path,
    *,
    dataset_path: Path,
    cv_splits: int,
    n_iter: int,
    random_state: int,
    test_size: float,
) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {
            "model": result.artifact.model_name,
            "cv_roc_auc": result.artifact.cv_roc_auc,
            **result.metrics.to_dict(),
        }
        rows.append(row)
        _write_coefficient_outputs(result.artifact, output_dir)

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    payload = {
        "dataset": str(dataset_path.resolve()),
        "cv_splits": cv_splits,
        "n_iter": n_iter,
        "random_state": random_state,
        "test_size": test_size,
        "results": rows,
        "best_params": {
            result.artifact.model_name: result.artifact.best_params
            for result in results
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_models(
    dataset_path: Path,
    output_dir: Path,
    model_names: list[ModelName],
    *,
    cv_splits: int = 5,
    n_iter: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    test_size: float = 0.25,
) -> list[TrainingResult]:
    """Train selected models against one shared, reproducible data split."""

    if not model_names:
        raise ValueError("At least one model must be selected")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Models may only be selected once")

    # Resolve native dependencies before beginning any potentially long fit.
    for model_name in model_names:
        _make_estimator(model_name, random_state)

    features, target = load_training_dataset(dataset_path)
    train_rows, test_rows = split_indices(
        target, test_size=test_size, random_state=random_state
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrainingResult] = []
    for model_name in model_names:
        artifact, metrics = train_model(
            model_name,
            features,
            target,
            train_rows,
            test_rows,
            cv_splits=cv_splits,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            test_size=test_size,
        )
        artifact_path = output_dir / "models" / f"{model_name}.joblib"
        save_artifact(artifact, artifact_path)
        results.append(TrainingResult(artifact, metrics, artifact_path))

    _write_run_outputs(
        results,
        output_dir,
        dataset_path=dataset_path,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
        test_size=test_size,
    )
    return results


def evaluate_saved_models(
    dataset_path: Path,
    artifact_paths: list[Path],
    output_dir: Path,
    *,
    plots_dir: Path | None = None,
) -> list[tuple[ModelArtifact, BinaryMetrics]]:
    """Re-evaluate saved artifacts using their original held-out rows."""

    features, target = load_training_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated: list[tuple[ModelArtifact, BinaryMetrics]] = []
    rows: list[dict[str, object]] = []

    for artifact_path in artifact_paths:
        artifact = load_artifact(artifact_path)
        metrics, probabilities = evaluate_artifact(artifact, features, target)
        evaluated.append((artifact, metrics))
        rows.append(
            {
                "model": artifact.model_name,
                "cv_roc_auc": artifact.cv_roc_auc,
                **metrics.to_dict(),
            }
        )
        if plots_dir is not None:
            from gutatlas.plotting import write_evaluation_plots

            test_target = target.iloc[list(artifact.test_indices)]
            write_evaluation_plots(
                artifact.model_name, test_target, probabilities, plots_dir
            )

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evaluated

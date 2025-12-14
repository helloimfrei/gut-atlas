"""
Train all models (XGBoost, LightGBM, Logistic Regression) and compare results.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_xgboost import train_xgboost
from train_lightgbm import train_lightgbm
from train_logreg import train_logreg


def train_all_models(
    data_path: str = "../data/processed/microbiomap/gi_binclass_training_set.parquet",
    model_dir: str = "../saved_models",
    cv_splits: int = 5,
    n_iter: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train all three models and display comparison.

    Parameters
    ----------
    data_path : str
        Path to processed training data
    model_dir : str
        Directory to save trained models
    cv_splits : int
        Number of cross-validation folds
    n_iter : int
        Number of Bayesian optimization iterations
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    results = []

    # Train XGBoost
    print("\n\n")
    xgb_tuner = train_xgboost(
        data_path=data_path,
        model_dir=model_dir,
        experiment_name="gi_xgboost",
        cv_splits=cv_splits,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    xgb_score = xgb_tuner.best_score()
    results.append({"Model": "XGBoost", "CV ROC AUC": xgb_score[1]})

    # Train LightGBM
    print("\n\n")
    lgbm_tuner = train_lightgbm(
        data_path=data_path,
        model_dir=model_dir,
        experiment_name="gi_lightgbm",
        cv_splits=cv_splits,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    lgbm_score = lgbm_tuner.best_score()
    results.append({"Model": "LightGBM", "CV ROC AUC": lgbm_score[1]})

    # Train Logistic Regression
    print("\n\n")
    logreg_tuner = train_logreg(
        data_path=data_path,
        model_dir=model_dir,
        experiment_name="gi_logreg",
        cv_splits=cv_splits,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    logreg_score = logreg_tuner.best_score()
    results.append({"Model": "Logistic Regression", "CV ROC AUC": logreg_score[1]})

    # Display comparison
    print("\n\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print()

    results_df = pd.DataFrame(results).sort_values("CV ROC AUC", ascending=False)
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]["Model"]
    best_score = results_df.iloc[0]["CV ROC AUC"]

    print()
    print(f"Best model: {best_model} (ROC AUC: {best_score:.4f})")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    train_all_models()

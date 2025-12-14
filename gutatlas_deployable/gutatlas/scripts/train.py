#!/usr/bin/env python3
"""
Train GI disease prediction models.

This script trains XGBoost, LightGBM, and Logistic Regression models
for binary classification of GI disease presence.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gutatlas.models import XGBBinClassTuner, LGBMBinClassTuner, LogRegBinClassTuner


def train_models(
    data_path: str,
    output_dir: str,
    cv_splits: int = 5,
    n_iter: int = 10,
    test_size: float = 0.25,
    random_state: int = 42,
    models_to_train: list = None
):
    """
    Train all models on the binary classification dataset.

    Args:
        data_path: Path to the training dataset parquet file
        output_dir: Directory to save trained models and parameters
        cv_splits: Number of cross-validation folds
        n_iter: Number of Bayesian optimization iterations
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        models_to_train: List of models to train ('xgboost', 'lightgbm', 'logreg')
    """
    if models_to_train is None:
        models_to_train = ['xgboost', 'lightgbm', 'logreg']

    # Create output directories
    models_dir = Path(output_dir) / "saved_models"
    params_dir = Path(output_dir) / "params"
    models_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    gi_training = pd.read_parquet(data_path)

    X = gi_training.drop(columns=["disease_present"])
    y = gi_training["disease_present"]

    print(f"Dataset shape: {X.shape}")
    print(f"Disease prevalence: {y.mean():.3f}")

    # Train-test split
    print(f"\nSplitting data (test_size={test_size}, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    results = {}

    # Train XGBoost
    if 'xgboost' in models_to_train:
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)

        xgb_tuner = XGBBinClassTuner(
            cv_splits=cv_splits,
            n_iter=n_iter,
            n_jobs=-1
        )

        print(f"Running Bayesian optimization ({n_iter} iterations, {cv_splits}-fold CV)...")
        xgb_tuner.fit(X_train, y_train)

        print(f"Best CV ROC AUC: {xgb_tuner.best_score_:.4f}")
        print(f"Best parameters: {xgb_tuner.best_params_}")

        # Save model and parameters
        model_path = models_dir / "gi_bin_class_xgboost_model.json"
        params_path = params_dir / "gi_bin_class_xgboost_params.json"

        xgb_tuner.save_model(str(models_dir), "gi_bin_class_xgboost_model.json")
        xgb_tuner.save_params(str(params_dir), "gi_bin_class_xgboost_params.json")

        print(f"Model saved to: {model_path}")
        print(f"Parameters saved to: {params_path}")

        results['xgboost'] = {
            'cv_roc_auc': xgb_tuner.best_score_,
            'best_params': xgb_tuner.best_params_
        }

    # Train LightGBM
    if 'lightgbm' in models_to_train:
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM")
        print("="*60)

        lgbm_tuner = LGBMBinClassTuner(
            cv_splits=cv_splits,
            n_iter=n_iter,
            n_jobs=-1
        )

        print(f"Running Bayesian optimization ({n_iter} iterations, {cv_splits}-fold CV)...")
        lgbm_tuner.fit(X_train, y_train)

        print(f"Best CV ROC AUC: {lgbm_tuner.best_score_:.4f}")
        print(f"Best parameters: {lgbm_tuner.best_params_}")

        # Save model and parameters
        model_path = models_dir / "gi_bin_class_lightgbm_model.json"
        params_path = params_dir / "gi_bin_class_lightgbm_params.json"

        lgbm_tuner.save_model(str(models_dir), "gi_bin_class_lightgbm_model.json")
        lgbm_tuner.save_params(str(params_dir), "gi_bin_class_lightgbm_params.json")

        print(f"Model saved to: {model_path}")
        print(f"Parameters saved to: {params_path}")

        results['lightgbm'] = {
            'cv_roc_auc': lgbm_tuner.best_score_,
            'best_params': lgbm_tuner.best_params_
        }

    # Train Logistic Regression
    if 'logreg' in models_to_train:
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)

        logreg_tuner = LogRegBinClassTuner(
            cv_splits=cv_splits,
            n_iter=n_iter,
            n_jobs=-1
        )

        print(f"Running Bayesian optimization ({n_iter} iterations, {cv_splits}-fold CV)...")
        logreg_tuner.fit(X_train, y_train)

        print(f"Best CV ROC AUC: {logreg_tuner.best_score_:.4f}")
        print(f"Best parameters: {logreg_tuner.best_params_}")

        # Save model and parameters
        model_path = models_dir / "gi_bin_class_logreg_model.pkl"
        params_path = params_dir / "gi_bin_class_logreg_params.json"

        logreg_tuner.save_model(str(models_dir), "gi_bin_class_logreg_model.pkl")
        logreg_tuner.save_params(str(params_dir), "gi_bin_class_logreg_params.json")

        print(f"Model saved to: {model_path}")
        print(f"Parameters saved to: {params_path}")

        results['logreg'] = {
            'cv_roc_auc': logreg_tuner.best_score_,
            'best_params': logreg_tuner.best_params_
        }

    # Save training summary
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training summary saved to: {summary_path}")

    for model_name, result in results.items():
        print(f"{model_name.upper()}: CV ROC AUC = {result['cv_roc_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train GI disease prediction models"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training dataset parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save trained models and parameters"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of Bayesian optimization iterations (default: 10)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of data for testing (default: 0.25)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        choices=['xgboost', 'lightgbm', 'logreg', 'all'],
        default=['all'],
        help="Models to train (default: all)"
    )

    args = parser.parse_args()

    # Handle 'all' option
    if 'all' in args.models:
        models_to_train = ['xgboost', 'lightgbm', 'logreg']
    else:
        models_to_train = args.models

    train_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        cv_splits=args.cv_splits,
        n_iter=args.n_iter,
        test_size=args.test_size,
        random_state=args.random_state,
        models_to_train=models_to_train
    )


if __name__ == "__main__":
    main()

"""
Train Logistic Regression binary classifier for GI disease prediction.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models import LogRegBinClassTuner, ModelManager
from models.visualizations import save_all_visualizations


def train_logreg(
    data_path: str = "../data/processed/microbiomap/gi_binclass_training_set.parquet",
    model_dir: str = "../saved_models",
    experiment_name: str = "gi_logreg",
    cv_splits: int = 5,
    n_iter: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train Logistic Regression model with Bayesian hyperparameter optimization.

    Parameters
    ----------
    data_path : str
        Path to processed training data
    model_dir : str
        Directory to save trained model
    experiment_name : str
        Name for the experiment
    cv_splits : int
        Number of cross-validation folds
    n_iter : int
        Number of Bayesian optimization iterations
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    random_state : int
        Random seed for reproducibility
    """
    print("=" * 60)
    print("Logistic Regression Training Pipeline")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading data from {data_path}...")
    gi_training = pd.read_parquet(data_path)
    X = gi_training.drop(columns=["disease_present"])
    y = gi_training["disease_present"]
    print(f"  ✓ Data shape: {X.shape}")
    print(f"  ✓ Target distribution: {y.value_counts().to_dict()}")

    # Train/test split
    print(f"\n[2/4] Creating train/test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    print(f"  ✓ Train size: {len(X_train)}")
    print(f"  ✓ Test size: {len(X_test)}")

    # Hyperparameter tuning
    print(f"\n[3/4] Starting Bayesian optimization...")
    print(f"  - CV folds: {cv_splits}")
    print(f"  - Iterations: {n_iter}")
    print(f"  - Scoring: roc_auc")
    print(f"  - Penalty: ElasticNet")

    tuner = LogRegBinClassTuner(
        cv_splits=cv_splits,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    tuner.fit(X_train, y_train)

    best_params = tuner.best_params()
    best_score = tuner.best_score()

    print(f"\n  ✓ Optimization complete!")
    print(f"  ✓ Best CV score ({best_score[0]}): {best_score[1]:.4f}")
    print(f"  ✓ Best parameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")

    # Save model
    print(f"\n[4/4] Saving model...")
    manager = ModelManager(
        model_type="logreg",
        model_dir=model_dir,
        experiment_name=experiment_name,
    )
    manager.save(tuner.best_estimator(), best_params)

    # Evaluate on test set
    print(f"\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    y_pred = tuner.best_estimator().predict(X_test)
    y_proba = tuner.best_estimator().predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  ROC AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]:5d}  |  FP: {cm[0, 1]:5d}")
    print(f"    FN: {cm[1, 0]:5d}  |  TP: {cm[1, 1]:5d}")

    # Save visualizations
    print(f"\n" + "=" * 60)
    print("Saving Visualizations")
    print("=" * 60)

    viz_paths = save_all_visualizations(
        model=tuner.best_estimator(),
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_type="logreg",
        experiment_name=experiment_name,
        output_dir=f"{model_dir}/figures",
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return tuner


if __name__ == "__main__":
    train_logreg()

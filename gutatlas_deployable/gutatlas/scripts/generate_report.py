#!/usr/bin/env python3
"""
Generate comprehensive metrics report for trained models.

This script evaluates all trained models and generates detailed metrics reports
including confusion matrices, ROC AUC scores, precision, recall, and specificity.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_models(models_dir: str):
    """Load all trained models."""
    models_dir = Path(models_dir)
    models = {}

    xgb_path = models_dir / "gi_bin_class_xgboost_model.json"
    if xgb_path.exists():
        models['XGBoost'] = xgb.Booster()
        models['XGBoost'].load_model(str(xgb_path))

    lgbm_path = models_dir / "gi_bin_class_lightgbm_model.json"
    if lgbm_path.exists():
        models['LightGBM'] = lgb.Booster(model_file=str(lgbm_path))

    logreg_path = models_dir / "gi_bin_class_logreg_model.pkl"
    if logreg_path.exists():
        models['Logistic Regression'] = joblib.load(logreg_path)

    return models


def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a single model and return metrics."""
    # Get predictions
    if model_name == 'XGBoost':
        dtest = xgb.DMatrix(X_test)
        y_proba = model.predict(dtest)
        y_pred = (y_proba >= 0.5).astype(int)

    elif model_name == 'LightGBM':
        y_proba = model.predict(X_test)
        y_pred = (y_proba >= 0.5).astype(int)

    elif model_name == 'Logistic Regression':
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)

    return {
        'test_roc_auc': roc_auc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }


def generate_report(
    data_path: str,
    models_dir: str,
    params_dir: str,
    output_dir: str,
    test_size: float = 0.25,
    random_state: int = 42
):
    """
    Generate comprehensive metrics report.

    Args:
        data_path: Path to training dataset parquet file
        models_dir: Directory containing trained models
        params_dir: Directory containing model parameters
        output_dir: Directory to save reports
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    gi_training = pd.read_parquet(data_path)

    X = gi_training.drop(columns=["disease_present"])
    y = gi_training["disease_present"]

    # Train-test split
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Load models
    print("Loading models...")
    models = load_models(models_dir)

    if not models:
        print("No models found. Please train models first.")
        return

    # Evaluate all models
    results = {}
    params_dir = Path(params_dir)

    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model_name, model, X_test, y_test)

        # Load CV ROC AUC from params file
        param_file_map = {
            'XGBoost': 'gi_bin_class_xgboost_params.json',
            'LightGBM': 'gi_bin_class_lightgbm_params.json',
            'Logistic Regression': 'gi_bin_class_logreg_params.json'
        }

        import json
        param_file = params_dir / param_file_map[model_name]
        if param_file.exists():
            with open(param_file, 'r') as f:
                params = json.load(f)
                metrics['cv_roc_auc'] = params.get('best_score', None)

        results[model_name] = metrics

        print(f"  Test ROC AUC: {metrics['test_roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)

    # Table 1: Overall Performance
    table1_data = []
    for model_name, metrics in results.items():
        table1_data.append({
            'Model': model_name,
            'CV ROC AUC': f"{metrics.get('cv_roc_auc', 0):.4f}" if metrics.get('cv_roc_auc') else '-',
            'Test ROC AUC': f"{metrics['test_roc_auc']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'Specificity': f"{metrics['specificity']:.4f}"
        })

    table1 = pd.DataFrame(table1_data)
    table1_path = output_dir / "table1_overall_performance.csv"
    table1.to_csv(table1_path, index=False)
    print(f"Table 1 saved to: {table1_path}")

    # Table 2: Confusion Matrices
    table2_data = []
    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']
        table2_data.append({
            'Model': model_name,
            'True Negatives (TN)': cm['tn'],
            'False Positives (FP)': cm['fp'],
            'False Negatives (FN)': cm['fn'],
            'True Positives (TP)': cm['tp'],
            'Total Test Samples': cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
        })

    table2 = pd.DataFrame(table2_data)
    table2_path = output_dir / "table2_confusion_matrices.csv"
    table2.to_csv(table2_path, index=False)
    print(f"Table 2 saved to: {table2_path}")

    # Table 3: Dataset Characteristics
    n_samples = len(gi_training)
    n_healthy = (y == 0).sum()
    n_diseased = (y == 1).sum()
    n_features = X.shape[1]

    table3_data = {
        'Characteristic': [
            'Total Samples',
            'Training Samples',
            'Test Samples',
            'Healthy Samples (y=0)',
            'Disease Samples (y=1)',
            'Class Balance Ratio',
            'Total Features'
        ],
        'Value': [
            f"{n_samples}",
            f"{len(X_train)} ({len(X_train)/n_samples*100:.1f}%)",
            f"{len(X_test)} ({len(X_test)/n_samples*100:.1f}%)",
            f"{n_healthy} ({n_healthy/n_samples*100:.1f}%)",
            f"{n_diseased} ({n_diseased/n_samples*100:.1f}%)",
            f"{n_diseased/n_samples:.3f}",
            f"{n_features} bacterial taxa"
        ]
    }

    table3 = pd.DataFrame(table3_data)
    table3_path = output_dir / "table3_dataset_characteristics.csv"
    table3.to_csv(table3_path, index=False)
    print(f"Table 3 saved to: {table3_path}")

    # Generate summary text
    summary_text = []
    summary_text.append("="*80)
    summary_text.append("COMPREHENSIVE METRICS SUMMARY FOR PROJECT REPORT")
    summary_text.append("="*80)
    summary_text.append("")
    summary_text.append("Dataset Statistics:")
    summary_text.append(f"  Total samples: {n_samples}")
    summary_text.append(f"  Total features: {n_features}")
    summary_text.append(f"  Training samples: {len(X_train)}")
    summary_text.append(f"  Test samples: {len(X_test)}")
    summary_text.append(f"  Healthy samples (y=0): {n_healthy}")
    summary_text.append(f"  Disease samples (y=1): {n_diseased}")
    summary_text.append(f"  Class balance (disease/total): {n_diseased/n_samples:.3f}")
    summary_text.append("")

    for model_name, metrics in results.items():
        summary_text.append("="*80)
        summary_text.append(f"{model_name} Metrics:")
        if metrics.get('cv_roc_auc'):
            summary_text.append(f"  CV ROC AUC: {metrics['cv_roc_auc']:.4f}")
        summary_text.append(f"  Test ROC AUC: {metrics['test_roc_auc']:.4f}")
        cm = metrics['confusion_matrix']
        summary_text.append(f"  Confusion Matrix:")
        summary_text.append(f"    TN={cm['tn']}, FP={cm['fp']}")
        summary_text.append(f"    FN={cm['fn']}, TP={cm['tp']}")
        summary_text.append(f"  Precision: {metrics['precision']:.4f}")
        summary_text.append(f"  Recall: {metrics['recall']:.4f}")
        summary_text.append(f"  Specificity: {metrics['specificity']:.4f}")
        summary_text.append("")

    summary_path = output_dir / "model_metrics_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_text))
    print(f"Summary saved to: {summary_path}")

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive metrics report"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training dataset parquet file"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--params-dir",
        type=str,
        required=True,
        help="Directory containing model parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save reports"
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

    args = parser.parse_args()

    generate_report(
        data_path=args.data_path,
        models_dir=args.models_dir,
        params_dir=args.params_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()

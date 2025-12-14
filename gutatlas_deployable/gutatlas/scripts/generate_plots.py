#!/usr/bin/env python3
"""
Generate visualizations for trained models.

This script creates confusion matrices, ROC curves, and SHAP importance plots
for all trained models.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gutatlas.models import plot_confusion_matrix, plot_roc_curve, plot_shap_importance


def load_models(models_dir: str):
    """Load all trained models."""
    models_dir = Path(models_dir)
    models = {}

    # Load XGBoost
    xgb_path = models_dir / "gi_bin_class_xgboost_model.json"
    if xgb_path.exists():
        models['xgboost'] = xgb.Booster()
        models['xgboost'].load_model(str(xgb_path))
        print(f"Loaded XGBoost model from {xgb_path}")

    # Load LightGBM
    lgbm_path = models_dir / "gi_bin_class_lightgbm_model.json"
    if lgbm_path.exists():
        models['lightgbm'] = lgb.Booster(model_file=str(lgbm_path))
        print(f"Loaded LightGBM model from {lgbm_path}")

    # Load Logistic Regression
    logreg_path = models_dir / "gi_bin_class_logreg_model.pkl"
    if logreg_path.exists():
        models['logreg'] = joblib.load(logreg_path)
        print(f"Loaded Logistic Regression model from {logreg_path}")

    return models


def generate_plots(
    data_path: str,
    models_dir: str,
    output_dir: str,
    test_size: float = 0.25,
    random_state: int = 42
):
    """
    Generate all visualizations for trained models.

    Args:
        data_path: Path to training dataset parquet file
        models_dir: Directory containing trained models
        output_dir: Directory to save plots
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

    # Train-test split (same as training)
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Load models
    models = load_models(models_dir)

    if not models:
        print("No models found. Please train models first.")
        return

    print(f"\nGenerating plots for {len(models)} models...")

    # Generate plots for each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Generating plots for {model_name.upper()}")
        print(f"{'='*60}")

        # Get predictions
        if model_name == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test)
            y_proba = model.predict(dtest)
            y_pred = (y_proba >= 0.5).astype(int)

        elif model_name == 'lightgbm':
            y_proba = model.predict(X_test)
            y_pred = (y_proba >= 0.5).astype(int)

        elif model_name == 'logreg':
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC: {roc_auc:.4f}")

        # Confusion Matrix
        print("  Generating confusion matrix...")
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(y_test, y_pred, display_labels=["Healthy", "Disease"])
        plt.title(f"{model_name.upper()} - Confusion Matrix")
        plt.tight_layout()
        confusion_path = output_dir / f"{model_name}_confusion.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved to: {confusion_path}")

        # ROC Curve
        print("  Generating ROC curve...")
        plt.figure(figsize=(8, 6))
        plot_roc_curve(y_test, y_proba, roc_auc)
        plt.title(f"{model_name.upper()} - ROC Curve (AUC = {roc_auc:.4f})")
        plt.tight_layout()
        roc_path = output_dir / f"{model_name}_roc_auc.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved to: {roc_path}")

        # SHAP Importance (for tree models)
        if model_name in ['xgboost', 'lightgbm']:
            print("  Generating SHAP importance plot...")
            plt.figure(figsize=(10, 8))

            if model_name == 'xgboost':
                dtest_shap = xgb.DMatrix(X_test)
                plot_shap_importance(model, dtest_shap, max_display=20, feature_names=X_test.columns.tolist())
            else:  # lightgbm
                plot_shap_importance(model, X_test, max_display=20, feature_names=X_test.columns.tolist())

            plt.title(f"{model_name.upper()} - SHAP Feature Importance")
            plt.tight_layout()
            shap_path = output_dir / f"{model_name}_shap.png"
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved to: {shap_path}")

        # Logistic Regression Coefficients
        if model_name == 'logreg':
            print("  Generating coefficient plots...")

            coef_df = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_[0]
            }).sort_values('coefficient')

            # Top protective features
            protective = coef_df.head(15)
            plt.figure(figsize=(10, 6))
            plt.barh(protective['feature'], protective['coefficient'], color='green', alpha=0.7)
            plt.xlabel('Coefficient (Protective)')
            plt.title('Top 15 Protective Genera')
            plt.tight_layout()
            protective_path = output_dir / f"{model_name}_protective_features.png"
            plt.savefig(protective_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved to: {protective_path}")

            # Save protective features to CSV
            protective.to_csv(output_dir / "gi_bin_class_logreg_protective_features.csv", index=False)

            # Top risk-enhancing features
            risk = coef_df.tail(15)
            plt.figure(figsize=(10, 6))
            plt.barh(risk['feature'], risk['coefficient'], color='red', alpha=0.7)
            plt.xlabel('Coefficient (Risk-Enhancing)')
            plt.title('Top 15 Risk-Enhancing Genera')
            plt.tight_layout()
            risk_path = output_dir / f"{model_name}_risk_features.png"
            plt.savefig(risk_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved to: {risk_path}")

            # Save risk features to CSV
            risk.to_csv(output_dir / "gi_bin_class_logreg_risk_features.csv", index=False)

    print("\n" + "="*60)
    print("PLOT GENERATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for trained models"
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
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save plots"
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

    generate_plots(
        data_path=args.data_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()

"""
Visualization utilities for model evaluation.

Generates and saves confusion matrices, ROC curves, and SHAP importance plots.
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    roc_auc_score,
)
import shap


def plot_confusion_matrix(
    y_test,
    y_pred,
    save_path=None,
    display_labels=["No Disease", "Disease"],
    title=None,
):
    """
    Plot and optionally save confusion matrix.

    Parameters
    ----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    save_path : str or Path, optional
        Path to save the plot
    display_labels : list, default=["No Disease", "Disease"]
        Labels for display
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The confusion matrix figure
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(y_test, y_proba, save_path=None, title=None):
    """
    Plot and optionally save ROC curve.

    Parameters
    ----------
    y_test : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    save_path : str or Path, optional
        Path to save the plot
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The ROC curve figure
    """
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, color="darkorange", linewidth=2)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.4f})", fontsize=14, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ ROC curve saved to {save_path}")

    return fig


def plot_shap_importance(
    model,
    X_test,
    model_type="xgboost",
    save_path=None,
    max_display=20,
    title=None,
):
    """
    Plot and optionally save SHAP feature importance.

    Parameters
    ----------
    model : trained model
        The trained model (XGBoost, LightGBM, or LogReg)
    X_test : pd.DataFrame or np.ndarray
        Test features for SHAP values
    model_type : str, default="xgboost"
        Type of model: "xgboost", "lightgbm", or "logreg"
    save_path : str or Path, optional
        Path to save the plot
    max_display : int, default=20
        Maximum number of features to display
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The SHAP importance figure
    """
    # Create appropriate explainer based on model type
    if model_type in ["xgboost", "lightgbm"]:
        # Tree-based models
        if model_type == "lightgbm" and hasattr(model, "booster_"):
            # LGBMClassifier has booster_ attribute
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
    elif model_type == "logreg":
        # Linear model
        explainer = shap.LinearExplainer(model, X_test)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Create summary plot
    fig = plt.figure(figsize=(10, 8))

    # For binary classification, shap_values might be 2D or 3D
    # If 3D (2 classes), use positive class (index 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap.summary_plot(
        shap_values,
        X_test,
        max_display=max_display,
        show=False,
    )

    if title:
        plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ SHAP importance saved to {save_path}")

    return fig


def save_all_visualizations(
    model,
    X_test,
    y_test,
    y_pred,
    y_proba,
    model_type,
    experiment_name,
    output_dir="../results/figures",
):
    """
    Generate and save all visualizations for a model.

    Parameters
    ----------
    model : trained model
        The trained model
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : array-like
        True test labels
    y_pred : array-like
        Predicted test labels
    y_proba : array-like
        Predicted probabilities for positive class
    model_type : str
        Type of model: "xgboost", "lightgbm", or "logreg"
    experiment_name : str
        Name of the experiment
    output_dir : str or Path
        Directory to save visualizations

    Returns
    -------
    dict
        Paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_display_name = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "logreg": "Logistic Regression",
    }.get(model_type, model_type)

    print(f"\nSaving visualizations for {model_display_name}...")

    paths = {}

    # Confusion Matrix
    cm_path = output_dir / f"{experiment_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=cm_path,
        title=f"{model_display_name} - Confusion Matrix",
    )
    plt.close()
    paths["confusion_matrix"] = cm_path

    # ROC Curve
    roc_path = output_dir / f"{experiment_name}_roc_curve.png"
    plot_roc_curve(
        y_test,
        y_proba,
        save_path=roc_path,
        title=f"{model_display_name} - ROC Curve",
    )
    plt.close()
    paths["roc_curve"] = roc_path

    # SHAP Importance (skip for now if issues, can be added later)
    try:
        shap_path = output_dir / f"{experiment_name}_shap_importance.png"
        plot_shap_importance(
            model,
            X_test,
            model_type=model_type,
            save_path=shap_path,
            max_display=20,
            title=f"{model_display_name} - SHAP Feature Importance",
        )
        plt.close()
        paths["shap_importance"] = shap_path
    except Exception as e:
        print(f"  ⚠ Could not generate SHAP plot: {e}")

    return paths

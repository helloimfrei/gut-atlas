"""Optional evaluation plots."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def write_evaluation_plots(
    model_name: str,
    target: pd.Series,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    """Write confusion-matrix and ROC plots for one model."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting dependencies are missing; run `uv sync --extra plots`"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = (probabilities >= 0.5).astype("int8")

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        target,
        predicted,
        display_labels=["Healthy", "Disease"],
        ax=axis,
        colorbar=False,
    )
    axis.set_title(f"{model_name.title()} confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_name}_confusion.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(target, probabilities, ax=axis)
    axis.set_title(f"{model_name.title()} ROC curve")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_name}_roc.png", dpi=200)
    plt.close(figure)

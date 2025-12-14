"""
Unified model management for saving, loading, and making predictions.

Supports XGBoost, LightGBM, and Logistic Regression models with a consistent interface.
"""

import json
from pathlib import Path
from typing import Literal, Union
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib


ModelType = Literal["xgboost", "lightgbm", "logreg"]


class ModelManager:
    """
    Unified interface for saving, loading, and using trained models.

    Parameters
    ----------
    model_type : ModelType
        Type of model: "xgboost", "lightgbm", or "logreg"
    model_dir : str or Path
        Directory where models and parameters are saved
    experiment_name : str
        Name of the experiment (used for file naming)
    """

    def __init__(
        self,
        model_type: ModelType,
        model_dir: Union[str, Path] = "../saved_models",
        experiment_name: str = "default"
    ):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.experiment_name = experiment_name
        self.model = None
        self.params = None

    @property
    def model_path(self) -> Path:
        """Get the model file path based on model type."""
        extensions = {
            "xgboost": ".json",
            "lightgbm": ".txt",
            "logreg": ".pkl"
        }
        ext = extensions[self.model_type]
        return self.model_dir / f"{self.experiment_name}_model{ext}"

    @property
    def params_path(self) -> Path:
        """Get the parameters file path."""
        return self.model_dir / f"{self.experiment_name}_params.json"

    def save(self, trained_model, params: dict):
        """
        Save a trained model and its parameters.

        Parameters
        ----------
        trained_model : XGBClassifier, LGBMClassifier, or LogisticRegression
            The trained model to save
        params : dict
            The hyperparameters used for training
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save model based on type
        if self.model_type == "xgboost":
            trained_model.save_model(str(self.model_path))
        elif self.model_type == "lightgbm":
            trained_model.booster_.save_model(str(self.model_path))
        elif self.model_type == "logreg":
            joblib.dump(trained_model, self.model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Save parameters
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=2)

        print(f"✓ Model saved to {self.model_path}")
        print(f"✓ Parameters saved to {self.params_path}")

    def load(self):
        """
        Load a saved model and its parameters.

        Returns
        -------
        self
            Returns self for method chaining
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model based on type
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
        elif self.model_type == "lightgbm":
            self.model = lgb.Booster(model_file=str(self.model_path))
        elif self.model_type == "logreg":
            self.model = joblib.load(self.model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load parameters if they exist
        if self.params_path.exists():
            with open(self.params_path, "r") as f:
                self.params = json.load(f)

        print(f"✓ Model loaded from {self.model_path}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")

        if self.model_type == "lightgbm":
            # LightGBM Booster returns probabilities, need to threshold
            proba = self.model.predict(X)
            return (proba > 0.5).astype(int)
        else:
            # XGBoost and LogReg have predict method
            return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Probability predictions for the positive class
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")

        if self.model_type == "lightgbm":
            # LightGBM Booster returns 1D array of probabilities
            return self.model.predict(X)
        else:
            # XGBoost and LogReg return 2D array, get positive class
            return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        """Get the model parameters."""
        if self.params is None:
            raise ValueError("No parameters loaded. Call load() first.")
        return self.params

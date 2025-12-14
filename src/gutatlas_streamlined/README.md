# Gutatlas Streamlined

A clean, organized ML pipeline for GI disease prediction from microbiome data.

## Directory Structure

```
gutatlas_streamlined/
├── data/               # Data processing pipelines
│   ├── __init__.py
│   └── process_gi_binary.py
├── models/             # Model tuning and management
│   ├── __init__.py
│   ├── tuners.py
│   └── model_manager.py
├── scripts/            # Training scripts
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── train_logreg.py
└── README.md
```

## Quick Start

### 1. Process Data

```python
from gutatlas_streamlined.data import process_gi_binary_dataset

# Process raw data into training-ready format
process_gi_binary_dataset(
    raw_data_dir="../data/interim/filtered_and_merged",
    output_path="../data/processed/microbiomap/gi_binclass_training_set.parquet"
)
```

Or run directly:
```bash
cd src/gutatlas_streamlined/data
python process_gi_binary.py
```

### 2. Train Models

#### XGBoost
```bash
cd src/gutatlas_streamlined/scripts
python train_xgboost.py
```

#### LightGBM
```bash
cd src/gutatlas_streamlined/scripts
python train_lightgbm.py
```

#### Logistic Regression
```bash
cd src/gutatlas_streamlined/scripts
python train_logreg.py
```

### 3. Load and Use Models

```python
from gutatlas_streamlined.models import ModelManager
import pandas as pd

# Load a trained model
manager = ModelManager(
    model_type="xgboost",  # or "lightgbm" or "logreg"
    model_dir="../saved_models",
    experiment_name="gi_xgboost"
)
manager.load()

# Make predictions
X_new = pd.read_parquet("new_data.parquet")
predictions = manager.predict(X_new)
probabilities = manager.predict_proba(X_new)
```

## Features

### Unified Model Management
- **ModelManager**: Single interface for all model types
- Consistent save/load across XGBoost, LightGBM, and Logistic Regression
- Automatic handling of model-specific formats (.json, .txt, .pkl)

### Data Processing
- **CLR transformation** for compositional data
- **Feature filtering** (removes shallow taxa, dead features)
- **Metadata integration** (BMI, age, region)
- Reproducible pipeline with clear logging

### Model Tuning
- **Bayesian optimization** with scikit-optimize
- **Stratified cross-validation** for imbalanced classes
- Sensible default search spaces for each model type
- Easy customization of hyperparameters

## Customization

### Custom Hyperparameter Search Space

```python
from gutatlas_streamlined.models import XGBBinClassTuner
from skopt.space import Real, Integer

custom_space = {
    "learning_rate": Real(0.01, 0.1),
    "max_depth": Integer(3, 10),
    "n_estimators": Integer(100, 500),
}

tuner = XGBBinClassTuner(
    cv_splits=5,
    n_iter=20,
    search_space=custom_space
)
```

### Custom Training Parameters

```python
from gutatlas_streamlined.scripts.train_xgboost import train_xgboost

train_xgboost(
    data_path="path/to/data.parquet",
    model_dir="path/to/save",
    experiment_name="my_experiment",
    cv_splits=10,
    n_iter=50,
    random_state=123
)
```

## Model Comparison

All training scripts output:
- Cross-validation ROC AUC score
- Best hyperparameters
- Test set performance (ROC AUC, accuracy, confusion matrix)

This makes it easy to compare models side-by-side.

## Visualizations

Each training script automatically generates and saves:
- **Confusion Matrix** - Classification accuracy breakdown
- **ROC Curve** - Model discrimination ability
- **SHAP Importance** - Feature importance for interpretability

Visualizations are saved to `{model_dir}/figures/` as high-resolution PNG files.

### Manual Visualization Generation

```python
from gutatlas_streamlined.models import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap_importance
)

# Generate individual plots
plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png")
plot_roc_curve(y_test, y_proba, save_path="roc_curve.png")
plot_shap_importance(model, X_test, model_type="xgboost", save_path="shap.png")
```

## Dependencies

- pandas, polars
- xgboost, lightgbm
- scikit-learn, scikit-optimize
- scikit-bio (for CLR transformation)
- pyjanitor

## Notes

- All scripts use stratified splits to handle class imbalance
- Models are automatically saved with their hyperparameters
- The ModelManager handles model-specific quirks (e.g., LightGBM probability outputs)
- Original files in `gutatlas/` are unchanged

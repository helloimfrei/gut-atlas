# GutAtlas: GI Disease Prediction from Gut Microbiome Composition

A streamlined machine learning package for predicting gastrointestinal disease risk using gut microbiome composition data.

## Overview

GutAtlas implements three machine learning models for binary classification of GI disease presence:
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **LightGBM**: Efficient gradient boosting for sparse, high-dimensional data
- **Logistic Regression**: Interpretable linear model with ElasticNet regularization

## Features

- **Dataset Creation**: Process raw Microbiomap data into analysis-ready datasets
- **Model Training**: Automated hyperparameter tuning with Bayesian optimization
- **Visualization**: Generate confusion matrices, ROC curves, and SHAP importance plots
- **Reporting**: Comprehensive metrics reports with performance tables

## Installation

```bash
# Clone the repository
cd gutatlas_deployable

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Build Dataset

Process raw microbiome data into a binary classification dataset:

```bash
python gutatlas/scripts/build_dataset.py \
    --taxon-path /path/to/taxonomic_table.csv \
    --metadata-path /path/to/sample_metadata.tsv \
    --tags-path /path/to/tags.tsv \
    --output-path ./data/gi_binclass_training_set.parquet
```

Optional flags:
- `--skip-batching`: Skip batch processing if batches already exist
- `--skip-regional`: Skip regional splitting if regional data already exists
- `--batch-size`: Number of rows per batch (default: 1000)

### 2. Train Models

Train all three models with cross-validation:

```bash
python gutatlas/scripts/train.py \
    --data-path ./data/gi_binclass_training_set.parquet \
    --output-dir ./ \
    --cv-splits 5 \
    --n-iter 10
```

Train specific models:

```bash
# Train only XGBoost and LightGBM
python gutatlas/scripts/train.py \
    --data-path ./data/gi_binclass_training_set.parquet \
    --output-dir ./ \
    --models xgboost lightgbm
```

### 3. Generate Visualizations

Create confusion matrices, ROC curves, and SHAP plots:

```bash
python gutatlas/scripts/generate_plots.py \
    --data-path ./data/gi_binclass_training_set.parquet \
    --models-dir ./saved_models \
    --output-dir ./plots
```

### 4. Generate Metrics Report

Create comprehensive metrics reports:

```bash
python gutatlas/scripts/generate_report.py \
    --data-path ./data/gi_binclass_training_set.parquet \
    --models-dir ./saved_models \
    --params-dir ./params \
    --output-dir ./
```

## Project Structure

```
gutatlas_deployable/
├── gutatlas/                  # Main package
│   ├── models/                # Model implementations
│   │   ├── xgboost.py         # XGBoost tuner
│   │   ├── lightgbm.py        # LightGBM tuner
│   │   ├── logreg.py          # Logistic regression tuner
│   │   └── metrics.py         # Evaluation metrics and plotting
│   ├── utils/                 # Utilities
│   │   └── constants.py       # GI tag definitions
│   ├── data.py                # Data processing functions
│   ├── features.py            # Feature engineering
│   └── scripts/               # Executable scripts
│       ├── build_dataset.py   # Dataset creation
│       ├── train.py           # Model training
│       ├── generate_plots.py  # Visualization generation
│       └── generate_report.py # Metrics reporting
├── data/                      # Data storage
├── saved_models/              # Trained models
├── params/                    # Hyperparameters
├── plots/                     # Generated visualizations
└── README.md                  # This file
```

## Data Processing Pipeline

The dataset creation pipeline performs the following steps:

1. **Batch Processing**: Load and normalize taxonomic data in batches
2. **TSS Normalization**: Total sum scaling to convert counts to relative abundances
3. **Regional Splitting**: Organize data by geographic region
4. **GI Tag Filtering**: Filter samples with GI disease tags
5. **Binary Labeling**: Map disease states to binary labels (0=healthy, 1=diseased)
6. **Deduplication**: Remove duplicate samples, prioritizing diseased records
7. **Feature Cleaning**: Remove dead features and shallow taxa (below family level)
8. **CLR Transformation**: Apply centered log-ratio transformation for compositional data

## Model Training Details

All models use:
- **5-fold stratified cross-validation**
- **Bayesian hyperparameter optimization** (10 iterations by default)
- **ROC AUC** as the optimization metric
- **Stratified train-test split** (75/25 by default)

### Hyperparameter Search Spaces

**XGBoost**:
- Learning rate: 0.01 - 0.3
- Max depth: 3 - 10
- Number of estimators: 100 - 1000
- Subsample ratio: 0.6 - 1.0
- Column sample ratio: 0.6 - 1.0
- L1/L2 regularization

**LightGBM**:
- Learning rate: 0.01 - 0.3
- Number of leaves: 20 - 150
- Number of estimators: 100 - 1000
- Subsample ratio: 0.6 - 1.0
- Column sample ratio: 0.6 - 1.0
- L1/L2 regularization

**Logistic Regression**:
- C (inverse regularization strength): 0.001 - 10.0
- L1 ratio (ElasticNet mixing): 0.0 - 1.0

## Output Files

### Trained Models
- `saved_models/gi_bin_class_xgboost_model.json`
- `saved_models/gi_bin_class_lightgbm_model.json`
- `saved_models/gi_bin_class_logreg_model.pkl`

### Hyperparameters
- `params/gi_bin_class_xgboost_params.json`
- `params/gi_bin_class_lightgbm_params.json`
- `params/gi_bin_class_logreg_params.json`

### Visualizations
- `plots/{model}_confusion.png` - Confusion matrices
- `plots/{model}_roc_auc.png` - ROC curves
- `plots/{model}_shap.png` - SHAP importance plots (tree models)
- `plots/logreg_protective_features.png` - Protective genera
- `plots/logreg_risk_features.png` - Risk-enhancing genera

### Reports
- `table1_overall_performance.csv` - Model performance metrics
- `table2_confusion_matrices.csv` - Confusion matrix values
- `table3_dataset_characteristics.csv` - Dataset statistics
- `model_metrics_summary.txt` - Comprehensive text report

## Expected Results

Based on the original analysis:

| Model | CV ROC AUC | Test ROC AUC | Precision | Recall | Specificity |
|-------|------------|--------------|-----------|--------|-------------|
| XGBoost | 0.8394 | 0.8353 | 0.7122 | 0.5734 | 0.8710 |
| LightGBM | 0.8340 | 0.8277 | 0.7134 | 0.5454 | 0.8780 |
| Logistic Regression | 0.7796 | 0.7775 | 0.6654 | 0.4971 | 0.8608 |

**Dataset**: 11,586 samples (7,442 healthy, 4,144 diseased) with 2,597 bacterial taxa features

## Usage as a Python Package

```python
from gutatlas.models import XGBBinClassTuner, LGBMBinClassTuner, LogRegBinClassTuner
from gutatlas.models import plot_confusion_matrix, plot_roc_curve, plot_shap_importance
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_parquet("./data/gi_binclass_training_set.parquet")
X = data.drop(columns=["disease_present"])
y = data["disease_present"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train XGBoost
xgb_tuner = XGBBinClassTuner(cv_splits=5, n_iter=10, n_jobs=-1)
xgb_tuner.fit(X_train, y_train)

# Save model
xgb_tuner.save_model("./saved_models", "my_xgboost_model.json")

# Evaluate
y_pred = xgb_tuner.predict(X_test)
y_proba = xgb_tuner.predict_proba(X_test)

# Visualize
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_proba[:, 1], roc_auc_score(y_test, y_proba[:, 1]))
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas >= 1.5.0
- polars >= 0.15.0
- scikit-learn >= 1.0.0
- xgboost >= 1.7.0
- lightgbm >= 3.3.0
- scikit-bio >= 0.5.0
- shap >= 0.41.0
- matplotlib >= 3.5.0
- bayes-opt >= 1.4.0

## Citation

If you use this package, please cite:

```
GutAtlas: Machine Learning for GI Disease Prediction from Gut Microbiome Composition
Built using data from the Human Microbiome Project (Microbiomap)
```

## License

This project is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository.

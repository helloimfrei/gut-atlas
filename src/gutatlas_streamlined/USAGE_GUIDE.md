# Gutatlas Streamlined - Usage Guide

## Complete Workflow

### Step 1: Process Data

```bash
cd src/gutatlas_streamlined/data
python process_gi_binary.py
```

This will:
1. Load filtered GI microbiome data
2. Remove dead features (zero abundance)
3. Create binary disease labels (0 = healthy, 1 = disease)
4. Remove duplicate samples
5. Filter to deep taxonomic levels (>= family)
6. Apply CLR transformation
7. Merge patient metadata (BMI, age, region)
8. Save to `data/processed/microbiomap/gi_binclass_training_set.parquet`

**Output**: Clean dataset ready for training

### Step 2: Train Models

You have three options:

#### Option A: Train Individual Models

```bash
cd src/gutatlas_streamlined/scripts

# Train XGBoost
python train_xgboost.py

# Train LightGBM
python train_lightgbm.py

# Train Logistic Regression
python train_logreg.py
```

#### Option B: Train All Models at Once

```bash
cd src/gutatlas_streamlined/scripts
python train_all.py
```

This will train all three models sequentially and display a comparison table.

**Output**:
- Trained models saved to `saved_models/`
- Hyperparameters saved as JSON files
- Visualizations saved to `saved_models/figures/`
  - Confusion matrix (PNG)
  - ROC curve (PNG)
  - SHAP feature importance (PNG)
- Console output with CV scores and test set performance

### Step 3: Load and Use Models

```python
from gutatlas_streamlined.models import ModelManager
import pandas as pd

# Load trained model
manager = ModelManager(
    model_type="xgboost",  # or "lightgbm" or "logreg"
    model_dir="saved_models",
    experiment_name="gi_xgboost"
)
manager.load()

# Load new data
X_new = pd.read_parquet("new_samples.parquet")

# Get predictions
binary_predictions = manager.predict(X_new)  # 0 or 1
probabilities = manager.predict_proba(X_new)  # probability scores

# View hyperparameters
params = manager.get_params()
print(params)
```

## Example: Full Pipeline from Scratch

```python
# 1. Process data
from gutatlas_streamlined.data import process_gi_binary_dataset

dataset = process_gi_binary_dataset(
    raw_data_dir="data/interim/filtered_and_merged",
    metadata_path="data/raw/microbiomap/sample_metadata.tsv",
    tags_path="data/raw/microbiomap/tags.tsv",
    output_path="data/processed/microbiomap/gi_binclass_training_set.parquet"
)

# 2. Train models
from gutatlas_streamlined.scripts.train_all import train_all_models

results = train_all_models(
    data_path="data/processed/microbiomap/gi_binclass_training_set.parquet",
    model_dir="saved_models",
    cv_splits=5,
    n_iter=20,  # More iterations for better optimization
    random_state=42
)

# 3. Load best model
from gutatlas_streamlined.models import ModelManager

best_model_name = results.iloc[0]["Model"].lower().replace(" ", "_")
manager = ModelManager(
    model_type=best_model_name.replace("_regression", "reg"),
    model_dir="saved_models",
    experiment_name=f"gi_{best_model_name}"
)
manager.load()

# 4. Make predictions on new data
predictions = manager.predict(X_new)
```

## Key Features

### 1. Unified Interface
All three models use the same interface:
- Same training script structure
- Same ModelManager for loading/inference
- Consistent output format

### 2. Hyperparameter Tuning
Bayesian optimization automatically finds good hyperparameters:
- Default search spaces are provided
- Customize by passing `search_space` parameter
- Results are saved for reproducibility

### 3. Model Comparison
Easy to compare models:
```python
from gutatlas_streamlined.scripts.train_all import train_all_models
results = train_all_models()
# Returns DataFrame with model names and CV scores
```

### 4. Reproducibility
- All scripts accept `random_state` parameter
- Hyperparameters are saved with models
- Data processing is deterministic

## Troubleshooting

### ImportError when running scripts
Make sure you're in the correct directory:
```bash
cd src/gutatlas_streamlined/scripts
python train_xgboost.py
```

Or use absolute imports by adding to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/gut-atlas/src"
python -m gutatlas_streamlined.scripts.train_xgboost
```

### Model file not found
Check that `model_dir` matches where you saved the model:
```python
manager = ModelManager(
    model_type="xgboost",
    model_dir="saved_models",  # Must match training script
    experiment_name="gi_xgboost"  # Must match training script
)
```

### Data processing fails
Ensure raw data exists:
- `data/interim/filtered_and_merged/gi_microbiomes_merged.parquet`
- `data/raw/microbiomap/sample_metadata.tsv`
- `data/raw/microbiomap/tags.tsv`

## Differences from Original Workflow

| Original | Streamlined |
|----------|-------------|
| Jupyter notebook (create_datasets.ipynb) | Python script (process_gi_binary.py) |
| Jupyter notebook (gi_binary_classification.ipynb) | Separate training scripts |
| Manual model saving | Unified ModelManager |
| Separate import for each model type | Single import: ModelManager |
| Model-specific loading code | Consistent load() method |

## Visualization Options

### Automatic Visualization (Default)

All training scripts automatically generate and save visualizations. No additional steps needed!

### Manual Visualization Generation

If you need to regenerate visualizations or create them for models trained without automatic saving:

```bash
cd src/gutatlas_streamlined/scripts

# Generate for all models
python generate_visualizations.py --model all

# Generate for specific model
python generate_visualizations.py --model xgboost
python generate_visualizations.py --model lightgbm
python generate_visualizations.py --model logreg
```

### Custom Visualization in Code

```python
from gutatlas_streamlined.models import (
    ModelManager,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap_importance,
)

# Load model and data
manager = ModelManager("xgboost", "saved_models", "gi_xgboost").load()
y_pred = manager.predict(X_test)
y_proba = manager.predict_proba(X_test)

# Generate individual plots with custom settings
plot_confusion_matrix(
    y_test, y_pred,
    save_path="my_confusion_matrix.png",
    title="Custom Confusion Matrix"
)

plot_roc_curve(
    y_test, y_proba,
    save_path="my_roc_curve.png",
    title="Custom ROC Curve"
)

plot_shap_importance(
    manager.model, X_test,
    model_type="xgboost",
    save_path="my_shap_plot.png",
    max_display=30,  # Show top 30 features
    title="Custom SHAP Plot"
)
```

## Next Steps

After training models, you can:
1. Review visualizations in `saved_models/figures/`
2. Evaluate on additional test sets
3. Analyze feature importance from SHAP plots
4. Compare ROC curves across models
5. Perform cross-dataset validation
6. Deploy best model to production

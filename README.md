# Gut Atlas

Gut Atlas is a reproducible research pipeline for predicting a binary
gastrointestinal (GI) disease label from gut microbiome composition. The
supported workflow replaces the original notebook execution order with an
installable Python package and command-line interface.

This is research software, not a clinical model or medical advice. The pooled
label combines heterogeneous GI conditions and source-study reporting styles;
the resulting associations should not be interpreted as causal effects.

The original project write-up is available as a
[final report](docs/final-project-report.pdf).

## Supported workflow

- Build the report-compatible GI binary dataset from the Microbiomap taxonomic
  table and tags.
- Apply total-sum scaling, remove shallow/dead taxa, replace compositional
  zeros, and apply centered log-ratio (CLR) transformation.
- Tune logistic regression, XGBoost, and LightGBM with Bayesian optimization
  and stratified cross-validation.
- Save versioned model artifacts with their feature schema and held-out row
  indices.
- Reproduce metrics, logistic-regression coefficient tables, confusion
  matrices, and ROC curves without running a notebook.

The BMI regression, GI multilabel, mental-health, external-dataset, and neural
network experiments remain in `notebooks/` as research provenance. They are not
part of the supported package because their analysis paths are incomplete or
not fully evaluated.

## Setup

The project uses Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

On macOS, XGBoost and LightGBM also need the OpenMP runtime supplied by Homebrew:

```bash
brew install libomp
```

Install the optional plotting dependency when needed:

```bash
uv sync --extra plots
```

The local checkout may have an ignored `data` symlink to the original iCloud
dataset. Commands accept explicit paths, so the package does not depend on that
machine-specific link.

## Usage

Validate the existing processed dataset:

```bash
uv run gut-atlas inspect \
  data/processed/microbiomap/gi_binclass_training_set.parquet
```

Rebuild it from the raw Microbiomap files:

```bash
uv run gut-atlas build-dataset \
  --taxon-table data/raw/microbiomap/taxonomic_table.csv \
  --tags data/raw/microbiomap/tags.tsv \
  --output data/processed/microbiomap/gi_binclass_training_set.parquet \
  --overwrite
```

Train all three supported models:

```bash
uv run gut-atlas train \
  data/processed/microbiomap/gi_binclass_training_set.parquet \
  --output-dir runs/gi-binary
```

Use one or more repeated `--model` flags to train a subset:

```bash
uv run gut-atlas train \
  data/processed/microbiomap/gi_binclass_training_set.parquet \
  --output-dir runs/logistic-smoke \
  --model logistic --cv-splits 2 --n-iter 1
```

Re-evaluate saved artifacts on the exact holdout used during training:

```bash
uv run gut-atlas evaluate \
  data/processed/microbiomap/gi_binclass_training_set.parquet \
  runs/gi-binary/models/*.joblib \
  --output-dir runs/gi-binary/evaluation \
  --plots-dir runs/gi-binary/plots
```

Each training run contains:

```text
runs/<name>/
├── models/*.joblib
├── features/logistic_coefficients.csv
├── features/logistic_protective.csv
├── features/logistic_risk_enhancing.csv
├── metrics.csv
└── run.json
```

Generated runs, artifacts, and the local data link are ignored by Git.

## Label compatibility

The binary label retains the original notebook rule: known healthy markers map
to `0`, other textual values map to `1`, and numeric strings are interpreted as
IBS-SSS scores with a threshold of 75. This deliberately preserves comparison
with the existing report. Changing the label ontology should be treated as a
new dataset version and followed by fresh validation.

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

The primary implementation is under `src/gutatlas/`; tests use small synthetic
fixtures and do not require the 4.8 GB research dataset.

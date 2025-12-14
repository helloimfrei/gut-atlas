"""
Generate visualizations for existing trained models.

Use this script to regenerate plots or create visualizations for models
trained without automatic visualization saving.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models import ModelManager
from models.visualizations import save_all_visualizations


def generate_visualizations_for_model(
    model_type="xgboost",
    experiment_name="gi_xgboost",
    data_path="../data/processed/microbiomap/gi_binclass_training_set.parquet",
    model_dir="../saved_models",
    output_dir="../saved_models/figures",
    random_state=42,
):
    """
    Load a trained model and generate visualizations.

    Parameters
    ----------
    model_type : str
        Type of model: "xgboost", "lightgbm", or "logreg"
    experiment_name : str
        Name of the experiment
    data_path : str
        Path to training data
    model_dir : str
        Directory where model is saved
    output_dir : str
        Directory to save visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("=" * 60)
    print(f"Generating Visualizations for {experiment_name}")
    print("=" * 60)

    # Load data
    print(f"\n[1/3] Loading data...")
    gi_training = pd.read_parquet(data_path)
    X = gi_training.drop(columns=["disease_present"])
    y = gi_training["disease_present"]

    # Create same train/test split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    print(f"  ✓ Test set size: {len(X_test)}")

    # Load model
    print(f"\n[2/3] Loading trained model...")
    manager = ModelManager(
        model_type=model_type,
        model_dir=model_dir,
        experiment_name=experiment_name,
    )
    manager.load()
    print(f"  ✓ Model loaded successfully")

    # Get predictions
    y_pred = manager.predict(X_test)
    y_proba = manager.predict_proba(X_test)

    # Generate visualizations
    print(f"\n[3/3] Generating visualizations...")
    viz_paths = save_all_visualizations(
        model=manager.model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_type=model_type,
        experiment_name=experiment_name,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print("=" * 60)
    print(f"\nSaved files:")
    for name, path in viz_paths.items():
        print(f"  • {name}: {path}")

    return viz_paths


def generate_all_visualizations(
    data_path="../data/processed/microbiomap/gi_binclass_training_set.parquet",
    model_dir="../saved_models",
    output_dir="../saved_models/figures",
    random_state=42,
):
    """
    Generate visualizations for all three models.

    Parameters
    ----------
    data_path : str
        Path to training data
    model_dir : str
        Directory where models are saved
    output_dir : str
        Directory to save visualizations
    random_state : int
        Random seed for reproducibility
    """
    models = [
        ("xgboost", "gi_xgboost"),
        ("lightgbm", "gi_lightgbm"),
        ("logreg", "gi_logreg"),
    ]

    for model_type, experiment_name in models:
        print("\n\n")
        try:
            generate_visualizations_for_model(
                model_type=model_type,
                experiment_name=experiment_name,
                data_path=data_path,
                model_dir=model_dir,
                output_dir=output_dir,
                random_state=random_state,
            )
        except FileNotFoundError as e:
            print(f"⚠ Skipping {experiment_name}: Model not found")
            continue
        except Exception as e:
            print(f"⚠ Error generating visualizations for {experiment_name}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualizations for trained models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "logreg", "all"],
        default="all",
        help="Which model to generate visualizations for",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (if not using default)",
    )

    args = parser.parse_args()

    if args.model == "all":
        generate_all_visualizations()
    else:
        experiment_name = args.experiment_name or f"gi_{args.model}"
        generate_visualizations_for_model(
            model_type=args.model, experiment_name=experiment_name
        )

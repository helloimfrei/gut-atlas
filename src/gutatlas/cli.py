"""Command-line interface for notebook-free Gut Atlas workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from gutatlas.dataset import build_gi_binary_dataset, summarize_dataset
from gutatlas.modeling import (
    SUPPORTED_MODELS,
    ModelName,
    evaluate_saved_models,
    train_models,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gut-atlas",
        description="Reproducible Gut Atlas data and modeling workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate and summarize a processed dataset"
    )
    inspect_parser.add_argument("dataset", type=_path)

    build_parser = subparsers.add_parser(
        "build-dataset", help="Build the GI binary dataset from Microbiomap files"
    )
    build_parser.add_argument("--taxon-table", type=_path, required=True)
    build_parser.add_argument("--tags", type=_path, required=True)
    build_parser.add_argument("--output", type=_path, required=True)
    build_parser.add_argument("--work-dir", type=_path)
    build_parser.add_argument("--batch-size", type=int, default=1_000)
    build_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser(
        "train", help="Tune and train one or more classifiers"
    )
    train_parser.add_argument("dataset", type=_path)
    train_parser.add_argument("--output-dir", type=_path, required=True)
    train_parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        action="append",
        dest="models",
        help="Model to train; repeat for multiple models (default: all)",
    )
    train_parser.add_argument("--cv-splits", type=int, default=5)
    train_parser.add_argument("--n-iter", type=int, default=10)
    train_parser.add_argument("--n-jobs", type=int, default=-1)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--test-size", type=float, default=0.25)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Re-evaluate saved models on their held-out rows"
    )
    evaluate_parser.add_argument("dataset", type=_path)
    evaluate_parser.add_argument("models", type=_path, nargs="+")
    evaluate_parser.add_argument("--output-dir", type=_path, required=True)
    evaluate_parser.add_argument("--plots-dir", type=_path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(summarize_dataset(args.dataset).to_dict(), indent=2))
            return 0

        if args.command == "build-dataset":
            summary = build_gi_binary_dataset(
                args.taxon_table,
                args.tags,
                args.output,
                batch_size=args.batch_size,
                work_dir=args.work_dir,
                overwrite=args.overwrite,
            )
            print(json.dumps(summary.to_dict(), indent=2))
            return 0

        if args.command == "train":
            selected = args.models or list(SUPPORTED_MODELS)
            results = train_models(
                args.dataset,
                args.output_dir,
                cast(list[ModelName], selected),
                cv_splits=args.cv_splits,
                n_iter=args.n_iter,
                n_jobs=args.n_jobs,
                random_state=args.random_state,
                test_size=args.test_size,
            )
            print(
                json.dumps(
                    [
                        {
                            "model": result.artifact.model_name,
                            "artifact": str(result.artifact_path),
                            "cv_roc_auc": result.artifact.cv_roc_auc,
                            **result.metrics.to_dict(),
                        }
                        for result in results
                    ],
                    indent=2,
                )
            )
            return 0

        if args.command == "evaluate":
            evaluated = evaluate_saved_models(
                args.dataset,
                args.models,
                args.output_dir,
                plots_dir=args.plots_dir,
            )
            print(
                json.dumps(
                    [
                        {"model": artifact.model_name, **metrics.to_dict()}
                        for artifact, metrics in evaluated
                    ],
                    indent=2,
                )
            )
            return 0
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

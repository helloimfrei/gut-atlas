"""Build and validate the GI binary-classification dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
import polars as pl

from gutatlas.constants import GI_TAGS
from gutatlas.features import (
    clean_feature_name,
    clr_transform,
    dead_features,
    is_shallow_taxon,
)
from gutatlas.labels import map_gi_status_binary

TARGET_COLUMN = "disease_present"


@dataclass(frozen=True)
class DatasetSummary:
    path: str
    samples: int
    features: int
    healthy: int
    diseased: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_columns(columns: list[str], required: set[str], source: Path) -> None:
    missing = required.difference(columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {joined}")


def _load_binary_labels(tags_path: Path) -> pl.DataFrame:
    tags = pl.read_csv(tags_path, separator="\t", schema_overrides={"value": pl.String})
    _require_columns(tags.columns, {"project", "srr", "tag", "value"}, tags_path)
    return (
        tags.filter(pl.col("tag").is_in(GI_TAGS))
        .with_columns(
            pl.concat_str(["project", "srr"], separator="_").alias("sample"),
            pl.col("value")
            .map_elements(map_gi_status_binary, return_dtype=pl.Int8)
            .alias(TARGET_COLUMN),
        )
        .group_by("sample")
        .agg(pl.col(TARGET_COLUMN).max())
    )


def _write_relevant_batches(
    taxon_table: Path,
    labels: pl.DataFrame,
    staging_dir: Path,
    batch_size: int,
) -> list[Path]:
    paths: list[Path] = []
    source_taxa: list[str] | None = None
    retained_taxa: list[str] | None = None

    batches = pl.scan_csv(taxon_table).collect_batches(chunk_size=batch_size)
    for batch in batches:
        _require_columns(batch.columns, {"sample"}, taxon_table)

        if source_taxa is None:
            sample_index = batch.columns.index("sample")
            source_taxa = batch.columns[sample_index + 1 :]
            retained_taxa = [
                column for column in source_taxa if not is_shallow_taxon(column)
            ]
            if not retained_taxa:
                raise ValueError("No family- or genus-level taxon columns were found")

        matched = batch.join(labels, on="sample", how="inner")
        if matched.is_empty():
            continue
        if source_taxa is None or retained_taxa is None:
            raise RuntimeError("Taxon columns were not initialized")

        total_column = "__total_reads"
        normalized = (
            matched.with_columns(
                pl.sum_horizontal([pl.col(column) for column in source_taxa]).alias(
                    total_column
                )
            )
            .filter(pl.col(total_column) > 0)
            .select(
                [
                    (pl.col(column) / pl.col(total_column)).alias(column)
                    for column in retained_taxa
                ]
                + [pl.col(TARGET_COLUMN)]
            )
        )
        if normalized.is_empty():
            continue
        path = staging_dir / f"batch_{len(paths):05d}.parquet"
        normalized.write_parquet(path)
        paths.append(path)

    if not paths:
        raise ValueError("No taxonomic samples matched the selected GI tags")
    return paths


def _finish_dataset(batch_paths: list[Path]) -> pd.DataFrame:
    frame = pl.scan_parquet([str(path) for path in batch_paths]).collect().to_pandas()
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    unused = dead_features(frame.loc[:, feature_columns])
    feature_columns = [column for column in feature_columns if column not in unused]

    cleaned_names = [clean_feature_name(column) for column in feature_columns]
    if len(set(cleaned_names)) != len(cleaned_names):
        raise ValueError("Feature cleaning produced duplicate taxon names")

    features = frame.loc[:, feature_columns].copy()
    features.columns = cleaned_names
    transformed = clr_transform(features)
    transformed[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8").to_numpy()
    return transformed


def build_gi_binary_dataset(
    taxon_table: Path,
    tags: Path,
    output: Path,
    *,
    batch_size: int = 1_000,
    work_dir: Path | None = None,
    overwrite: bool = False,
) -> DatasetSummary:
    """Build the report-compatible GI binary dataset from Microbiomap inputs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for source in (taxon_table, tags):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_binary_labels(tags)
    with TemporaryDirectory(prefix="gut-atlas-", dir=work_dir) as temporary:
        batch_paths = _write_relevant_batches(
            taxon_table, labels, Path(temporary), batch_size
        )
        result = _finish_dataset(batch_paths)
        result.to_parquet(output, index=False)

    return summarize_dataset(output)


def load_training_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a validated training dataset."""

    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain a {TARGET_COLUMN!r} column")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")

    features = frame.drop(columns=[TARGET_COLUMN])
    if features.empty:
        raise ValueError("Dataset does not contain any features")
    nonnumeric = features.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise ValueError(f"Non-numeric features: {', '.join(map(str, nonnumeric))}")
    values = features.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Dataset features contain missing or infinite values")

    target = cast(pd.Series, frame[TARGET_COLUMN])
    if bool(target.isna().any()) or set(target.unique()) != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN!r} must contain both binary classes 0 and 1")
    return features, cast(pd.Series, target.astype("int8"))


def summarize_dataset(path: Path) -> DatasetSummary:
    features, target = load_training_dataset(path)
    return DatasetSummary(
        path=str(path.resolve()),
        samples=len(target),
        features=features.shape[1],
        healthy=int((target == 0).sum()),
        diseased=int((target == 1).sum()),
    )

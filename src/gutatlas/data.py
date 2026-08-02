"""Compatibility exports for code developed in the research notebooks."""

from collections.abc import Collection
from pathlib import Path

import polars as pl

from gutatlas.labels import map_gi_status_binary, normalize_multilabel_gi_tags

__all__ = [
    "filter_by_tag",
    "map_gi_status_binary",
    "normalize_multilabel_gi_tags",
]


def filter_by_tag(batch_dir: str | Path, tags: Collection[str]) -> pl.DataFrame:
    """Load all parquet batches in a directory and keep matching tag rows."""

    pattern = str(Path(batch_dir) / "*.parquet")
    return pl.scan_parquet(pattern).filter(pl.col("tag").is_in(list(tags))).collect()

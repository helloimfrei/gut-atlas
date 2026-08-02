"""Feature cleaning and compositional transforms."""

from __future__ import annotations

import re
from typing import cast

import numpy as np
import pandas as pd

_NON_IDENTIFIER = re.compile(r"[^0-9a-zA-Z_]")


def clean_feature_name(name: object) -> str:
    """Convert a source taxon name to a stable dataframe column name."""

    return _NON_IDENTIFIER.sub("_", str(name).replace(".", "_")).casefold()


def is_shallow_taxon(name: str) -> bool:
    """Return whether a taxon is unidentified above the family level."""

    levels = name.replace("_", ".").split(".")
    return any(level.casefold() == "na" for level in levels[:5])


def dead_features(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns whose total abundance is zero."""

    totals = cast(pd.Series, frame.sum(axis=0, numeric_only=True))
    return [str(column) for column, total in totals.items() if float(total) == 0.0]


def clr_transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply multiplicative zero replacement and centered log-ratio scaling."""

    values = frame.to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("CLR transformation requires at least one feature")
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains non-finite values")
    if (values < 0).any():
        raise ValueError("Compositional features cannot be negative")

    row_totals = values.sum(axis=1, keepdims=True)
    if (row_totals <= 0).any():
        raise ValueError("Compositional rows must have a positive total")
    values /= row_totals

    zero_mask = values == 0
    zero_counts = zero_mask.sum(axis=1, keepdims=True)
    delta = 1.0 / values.shape[1] ** 2
    remaining_mass = 1.0 - zero_counts * delta
    if (remaining_mass <= 0).any():
        raise ValueError("Too many zero values for multiplicative replacement")

    values *= remaining_mass
    values[zero_mask] = delta
    logged = np.log(values)
    transformed = logged - logged.mean(axis=1, keepdims=True)
    return pd.DataFrame(transformed, columns=frame.columns, index=frame.index)


def short_taxon_name(name: str) -> str:
    """Return the family and genus portion of a cleaned taxon name."""

    levels = name.split("_")
    return "_".join(levels[-2:]) if len(levels) >= 2 else name

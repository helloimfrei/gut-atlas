"""GI disease label normalization."""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import cast

import pandas as pd

from gutatlas.constants import HEALTHY_MARKERS

CONTAINER_TAGS: frozenset[str] = frozenset(
    {
        "gastrointest_disord",
        "pm_gastro_problems",
        "pm_gastro_problems_other",
        "pm_gastro_problems_unspecified",
    }
)

NEGATIVE_TOKENS = HEALTHY_MARKERS | {"n", "missing"}

DISEASE_SYNONYMS: dict[str, str] = {
    "irritable bowel syndrome": "IBS",
    "ibs": "IBS",
    "mild irritable bowel syndrome": "IBS",
    "yes.ibs": "IBS",
    "ulcerative colitis": "Ulcerative colitis",
    "uclerative_colitis": "Ulcerative colitis",
    "uc": "Ulcerative colitis",
    "crohn": "Crohn's disease",
    "crohn's disease": "Crohn's disease",
    "crohns": "Crohn's disease",
    "cd": "Crohn's disease",
    "crohn (remission)": "Crohn's disease",
    "ileal crohn's disease": "Crohn's disease",
    "colonic crohn's disease": "Crohn's disease",
    "ileal and colonic crohn's disease": "Crohn's disease",
    "functional dyspepsia": "Functional dyspepsia",
    "functional_dyspepsia": "Functional dyspepsia",
    "gastritis": "Gastritis",
    "obesity": "Obesity",
    "microcolitis": "Microcolitis",
    "colitis": "Colitis",
    "gi_ca": "GI cancer",
    "cancer": "GI cancer",
    "gastrointestinal cancer": "GI cancer",
    "advanced gastric cancer": "GI cancer",
    "early gastric cancer": "GI cancer",
    "clostridium difficile infection": "C. difficile infection",
    "necrotizing enterocolitis": "Necrotizing enterocolitis",
    "pouchitis": "Pouchitis",
    "sibo": "SIBO",
    "acid_reflux": "GERD",
    "gerd": "GERD",
}

_MULTIVALUE_SEPARATOR = re.compile(r"[;,/|]| and | or ", flags=re.IGNORECASE)


def map_gi_status_binary(value: object) -> int:
    """Map the source notebook's heterogeneous GI values to zero or one.

    Numeric strings are treated as IBS-SSS scores, retaining the notebook's
    threshold of 75 so rebuilt datasets remain comparable with the report.
    """

    if value is None:
        return 0
    if isinstance(value, Real) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return 0
        numeric_value = float(value)
        value = str(int(numeric_value)) if numeric_value.is_integer() else str(value)

    normalized = str(value).strip().casefold()
    if normalized.isdigit():
        return int(int(normalized) >= 75)
    return int(normalized not in HEALTHY_MARKERS)


def _canonical_disease(value: object) -> str | None:
    normalized = str(value).strip().casefold()
    if not normalized:
        return None
    if normalized in NEGATIVE_TOKENS:
        return "__negative__"
    if normalized in DISEASE_SYNONYMS:
        return DISEASE_SYNONYMS[normalized]
    if "crohn" in normalized or normalized == "cd":
        return "Crohn's disease"
    if "ulcerative colitis" in normalized or normalized == "uc":
        return "Ulcerative colitis"
    if "irritable bowel" in normalized or normalized == "ibs":
        return "IBS"
    if "reflux" in normalized or normalized == "gerd":
        return "GERD"
    if "sibo" in normalized:
        return "SIBO"
    if "gastritis" in normalized:
        return "Gastritis"
    if "dyspepsia" in normalized:
        return "Functional dyspepsia"
    if "cancer" in normalized or "gi_ca" in normalized:
        return "GI cancer"
    if "pouchitis" in normalized:
        return "Pouchitis"
    if "clostridium difficile" in normalized:
        return "C. difficile infection"
    if "necrotizing enterocolitis" in normalized:
        return "Necrotizing enterocolitis"
    return None


def _is_missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, Real) and math.isnan(float(value)))
    )


def _normalize_container(sample: str, value: object) -> list[tuple[str, str, int]]:
    if _is_missing(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []

    rows: list[tuple[str, str, int]] = []
    tokens = [part for part in _MULTIVALUE_SEPARATOR.split(raw) if part.strip()]
    for token in tokens or [raw]:
        disease = _canonical_disease(token)
        if disease == "__negative__":
            rows.append((sample, "GI_other", 0))
        elif disease is None:
            rows.append((sample, "GI_other", 1))
        else:
            rows.append((sample, disease, 1))
    return rows


def _normalize_regular(
    sample: str, tag: str, value: object
) -> list[tuple[str, str, int]]:
    normalized_value = "" if _is_missing(value) else str(value).strip().casefold()
    if tag.upper() == "IBS_SSS":
        try:
            return [(sample, "IBS", int(int(normalized_value) >= 75))]
        except ValueError:
            return [(sample, "IBS", 0)]

    disease = _canonical_disease(tag)
    if disease is None or disease == "__negative__":
        return []
    is_present = normalized_value not in NEGATIVE_TOKENS and normalized_value != "0"
    return [(sample, disease, int(is_present))]


def normalize_multilabel_gi_tags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical, deduplicated disease labels for each sample."""

    required = {"sample", "tag", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    rows: list[tuple[str, str, int]] = []
    for sample, tag, value in frame.loc[:, ["sample", "tag", "value"]].itertuples(
        index=False, name=None
    ):
        sample_text = str(sample)
        tag_text = str(tag)
        if tag_text in CONTAINER_TAGS:
            rows.extend(_normalize_container(sample_text, value))
        else:
            rows.extend(_normalize_regular(sample_text, tag_text, value))

    if not rows:
        return pd.DataFrame(columns=["sample", "tag", "value"])

    normalized = pd.DataFrame(rows, columns=["sample", "tag", "value"])
    normalized["value"] = normalized["value"].astype("int8")
    grouped = normalized.groupby(["sample", "tag"], as_index=False)["value"].max()
    return cast(pd.DataFrame, grouped)

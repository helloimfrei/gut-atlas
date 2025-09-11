import pandas as pd
import polars as pl
from pathlib import Path
import re

def filter_by_tag(batch_dir, tag_list):
    """"
    Filter a batch of parquet files by a list of tags.

    Returns a concatenated Polars DataFrame
    """
    lf = pl.scan_parquet(str(Path(batch_dir) / "*.parquet"), glob=True)
    filtered = lf.filter(pl.col("tag").is_in(tag_list))
    return filtered.collect() 

#chat saved me an immense amount of time on the below functions
# #cleaning these tags/values would not have been very fun to do myself

# cleaning up the values column and converting to binary diseased(1) or not diseased (0)
def map_gi_status_binary(value):
    """
    Map GI-related values (categorical or numeric IBS-SSS) to binary 0/1.
    
    Args:
        value (str or int): raw tag value from dataset
    
    Returns:
        int: 0 = healthy/none, 1 = any GI issue
    """
    if value is None:
        return 0
    
    val = str(value).strip().lower()
    
    # ---- Handle numeric IBS-SSS scores ----
    if val.isdigit():
        score = int(val)
        return 1 if score >= 75 else 0
    
    # ---- Known healthy/negative markers ----
    healthy_markers = {
        "i do not have this condition", "none", "normal", "healthy",
        "no", "no diagnosed disorders", "not applicable",
        "not_applicable", "na", "unk", "not provided", "unspecified",
        "not collected", "control", "labcontrol test", "hc", "healthy_control"
    }
    
    if val in healthy_markers:
        return 0
    
    # ---- Otherwise assume positive GI condition ----
    return 1

##
#cleaning gi tags/valus for multilabel classification 

CONTAINER_TAGS = {
    "gastrointest_disord",
    "pm_gastro_problems",
    "pm_gastro_problems_other",
    "pm_gastro_problems_unspecified",
}

NEG_TOKENS = {
    "none","no","n","healthy","healthy_control","control",
    "no diagnosed disorders","not applicable","not_applicable",
    "not collected","not provided","unspecified","missing","unk"
}

# Canonical disease map (expand as you encounter more)
DISEASE_SYNONYMS = {
    # IBS
    "irritable bowel syndrome": "IBS",
    "ibs": "IBS",
    "mild irritable bowel syndrome": "IBS",
    "yes.ibs": "IBS",

    # Ulcerative colitis
    "ulcerative colitis": "Ulcerative colitis",
    "uclerative_colitis": "Ulcerative colitis",
    "uc": "Ulcerative colitis",

    # Crohn's disease
    "crohn": "Crohn's disease",
    "crohn's disease": "Crohn's disease",
    "crohns": "Crohn's disease",
    "cd": "Crohn's disease",
    "crohn (remission)": "Crohn's disease",
    "ileal crohn's disease": "Crohn's disease",
    "colonic crohn's disease": "Crohn's disease",
    "ileal and colonic crohn's disease": "Crohn's disease",

    # Other GI diagnoses
    "functional dyspepsia": "Functional dyspepsia",
    "functional_dyspepsia": "Functional dyspepsia",
    "gastritis": "Gastritis",
    "obesity": "Obesity",
    "microcolitis": "Microcolitis",
    "colitis": "Colitis",

    # Cancer
    "gi_ca": "GI cancer",
    "cancer": "GI cancer",
    "gastrointestinal cancer": "GI cancer",
    "advanced gastric cancer": "GI cancer",
    "early gastric cancer": "GI cancer",

    # Infections / other
    "clostridium difficile infection": "C. difficile infection",
    "necrotizing enterocolitis": "Necrotizing enterocolitis",

    # Related
    "pouchitis": "Pouchitis",
    "sibo": "SIBO",
    "acid_reflux": "GERD",
    "gerd": "GERD",
}

# Optional: splitter for multi-valued strings (if they appear)
SPLIT_RE = re.compile(r"[;,/|]| and | or ", flags=re.IGNORECASE)


# ----------------------------
# Helpers
# ----------------------------
def _canon_token(text: str):
    """
    Return canonical disease string for a raw token,
    '__NEG__' for negatives, or None if unknown.
    """
    t = str(text).strip().lower()
    if not t:
        return None
    if t in NEG_TOKENS:
        return "__NEG__"
    if t in DISEASE_SYNONYMS:
        return DISEASE_SYNONYMS[t]

    # Light heuristics
    if "crohn" in t or t == "cd":
        return "Crohn's disease"
    if "ulcerative colitis" in t or t == "uc":
        return "Ulcerative colitis"
    if "irritable bowel" in t or t == "ibs":
        return "IBS"
    if "reflux" in t or t == "gerd":
        return "GERD"
    if "sibo" in t:
        return "SIBO"
    if "gastritis" in t:
        return "Gastritis"
    if "dyspepsia" in t:
        return "Functional dyspepsia"
    if "cancer" in t or "gi_ca" in t:
        return "GI cancer"
    if "pouchitis" in t:
        return "Pouchitis"
    if "clostridium difficile" in t:
        return "C. difficile infection"
    if "necrotizing enterocolitis" in t:
        return "Necrotizing enterocolitis"
    return None


def _normalize_container(sample: str, value) -> list[tuple[str, str, int]]:
    """
    For rows where tag is a container (e.g., 'gastrointest_disord') and
    the disease appears in 'value'.
    Returns list of (sample, canonical_tag, 0/1).
    """
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    tokens = [p for p in SPLIT_RE.split(raw) if p.strip()] or [raw]

    out = []
    for tok in tokens:
        m = _canon_token(tok)
        if m == "__NEG__":
            out.append((sample, "GI_other", 0))
        elif m:
            out.append((sample, m, 1))
        else:
            # unknown disease string — keep as GI_other present (tweak if you prefer dropping)
            out.append((sample, "GI_other", 1))
    return out


def _normalize_regular(sample: str, tag: str, value) -> list[tuple[str, str, int]]:
    """
    For rows where 'tag' is already a disease/specific indicator.
    Handles IBS_SSS binning, negatives, numeric, etc.
    """
    t = str(tag)
    v = "" if pd.isna(value) else str(value).strip().lower()

    # IBS-SSS numeric scale → binarize at 75
    if t.upper() == "IBS_SSS":
        try:
            score = int(v)
            return [(sample, "IBS", 1 if score >= 75 else 0)]
        except Exception:
            return [(sample, "IBS", 0)]

    # canonicalize the tag itself
    m_tag = _canon_token(t)
    if m_tag is None:
        return []  # unknown tag → skip (or map to GI_other if desired)

    # value-based logic (very permissive towards presence)
    if v in NEG_TOKENS or v == "0":
        y = 0
    elif v in {
        "1","yes","present",
        "self-diagnosed",
        "diagnosed by a medical professional (doctor, physician assistant)",
        "diagnosed by an alternative medicine practitioner"
    }:
        y = 1
    elif v.isdigit():
        y = 1
    else:
        y = 1

    return [(sample, m_tag, y)]


# ----------------------------
# Public function
# ----------------------------

def normalize_multilabel_gi_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: ['sample', 'tag', 'value']
    Output df columns: ['sample', 'tag', 'value'] with:
      - tag = canonical disease label
      - value = 0/1 presence
    Deduplicates per (sample, tag) by taking max (any positive wins).
    """
    rows: list[tuple[str, str, int]] = []

    for _, r in df[["sample", "tag", "value"]].iterrows():
        sample, tag, value = r["sample"], str(r["tag"]), r["value"]

        if tag in CONTAINER_TAGS:
            rows.extend(_normalize_container(sample, value))
        else:
            rows.extend(_normalize_regular(sample, tag, value))

    if not rows:
        return pd.DataFrame(columns=["sample", "tag", "value"])

    out = pd.DataFrame(rows, columns=["sample", "tag", "value"]).dropna()
    out["value"] = out["value"].astype(int)

    # If multiple rows per (sample, tag), keep any positive
    out = out.groupby(["sample", "tag"], as_index=False)["value"].max()
    return out

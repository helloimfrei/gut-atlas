from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gutatlas.dataset import TARGET_COLUMN, build_gi_binary_dataset


def test_build_gi_binary_dataset_from_raw_tables(tmp_path: Path) -> None:
    taxon_table = tmp_path / "taxa.csv"
    tags = tmp_path / "tags.tsv"
    output = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "row": [0, 1, 2],
            "sample": ["P_s1", "P_s2", "P_s3"],
            "Bacteria.P.C.O.F.G1": [10, 0, 5],
            "Bacteria.P.C.O.F.G2": [10, 10, 5],
            "Bacteria.P.C.O.NA.G3": [5, 5, 5],
            "Bacteria.P.C.O.F.G_dead": [0, 0, 0],
        }
    ).to_csv(taxon_table, index=False)
    pd.DataFrame(
        {
            "project": ["P", "P", "P", "P"],
            "srr": ["s1", "s2", "s2", "s3"],
            "srs": ["r1", "r2", "r2", "r3"],
            "tag": ["ibs", "ibs", "not_gi", "crohns"],
            "value": ["none", "diagnosed", "none", "diagnosed"],
        }
    ).to_csv(tags, sep="\t", index=False)

    summary = build_gi_binary_dataset(taxon_table, tags, output, batch_size=2)
    result = pd.read_parquet(output)

    assert summary.samples == 3
    assert summary.features == 2
    assert summary.healthy == 1
    assert summary.diseased == 2
    assert result.shape == (3, 3)
    assert set(result[TARGET_COLUMN]) == {0, 1}
    assert np.isfinite(result.drop(columns=[TARGET_COLUMN]).to_numpy()).all()
    assert result.drop(columns=[TARGET_COLUMN]).sum(axis=1).to_numpy() == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12
    )

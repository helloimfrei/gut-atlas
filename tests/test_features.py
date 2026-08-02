import numpy as np
import pandas as pd
import pytest

from gutatlas.features import clean_feature_name, clr_transform, is_shallow_taxon


def test_clean_feature_name() -> None:
    assert (
        clean_feature_name("Bacteria.Family.Genus group")
        == "bacteria_family_genus_group"
    )


def test_is_shallow_taxon_only_checks_through_family() -> None:
    assert is_shallow_taxon("Bacteria.Phylum.Class.Order.NA.Genus")
    assert not is_shallow_taxon("Bacteria.Phylum.Class.Order.Family.NA")


def test_clr_transform_is_finite_and_centered() -> None:
    frame = pd.DataFrame({"a": [1.0, 0.0], "b": [1.0, 2.0], "c": [2.0, 1.0]})

    result = clr_transform(frame)

    assert np.isfinite(result.to_numpy()).all()
    assert result.sum(axis=1).to_numpy() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_clr_transform_rejects_empty_compositions() -> None:
    with pytest.raises(ValueError, match="positive total"):
        clr_transform(pd.DataFrame({"a": [0.0], "b": [0.0]}))

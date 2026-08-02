import pandas as pd
import pytest

from gutatlas.labels import map_gi_status_binary, normalize_multilabel_gi_tags


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("healthy", 0),
        ("not provided", 0),
        ("74", 0),
        ("75", 1),
        ("Crohn's disease", 1),
    ],
)
def test_map_gi_status_binary_preserves_notebook_rules(
    value: object, expected: int
) -> None:
    assert map_gi_status_binary(value) == expected


def test_normalize_multilabel_tags_deduplicates_positive_labels() -> None:
    frame = pd.DataFrame(
        {
            "sample": ["a", "a", "b", "b"],
            "tag": ["ibs", "ibs", "gastrointest_disord", "gastrointest_disord"],
            "value": ["no", "yes", "Crohn's disease", "none"],
        }
    )

    result = normalize_multilabel_gi_tags(frame)

    assert (
        result.loc[(result["sample"] == "a") & (result["tag"] == "IBS"), "value"].item()
        == 1
    )
    assert (
        result.loc[
            (result["sample"] == "b") & (result["tag"] == "Crohn's disease"),
            "value",
        ].item()
        == 1
    )

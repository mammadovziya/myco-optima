"""Tests for safe downloadable result serialization."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from myco_optima.exports import (
    dataframe_to_safe_csv,
    neutralise_spreadsheet_formula,
    strict_json_dumps,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=1+1", "'=1+1"),
        ("+cmd", "'+cmd"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1:A2)", "'@SUM(A1:A2)"),
        ('  =HYPERLINK("https://example.com")', '\'  =HYPERLINK("https://example.com")'),
        ("\t=cmd", "'\t=cmd"),
        ("BIOMASS", "BIOMASS"),
        (4.2, 4.2),
    ],
)
def test_formula_like_text_is_neutralised(value, expected) -> None:
    assert neutralise_spreadsheet_formula(value) == expected


def test_safe_csv_does_not_mutate_source_and_preserves_numeric_negatives() -> None:
    source = pd.DataFrame(
        {
            "Reaction": ["=malicious", "normal"],
            "Name": ["+formula", "exchange"],
            "Flux": [-3.5, 2.0],
        }
    )

    csv_text = dataframe_to_safe_csv(source).decode("utf-8")

    assert "'=malicious" in csv_text
    assert "'+formula" in csv_text
    assert "-3.5" in csv_text
    assert source.iloc[0]["Reaction"] == "=malicious"


def test_strict_json_is_valid_and_rejects_non_finite_numbers() -> None:
    encoded = strict_json_dumps({"status": "optimal", "value": 1.25})
    assert json.loads(encoded) == {"status": "optimal", "value": 1.25}

    with pytest.raises(ValueError, match="Out of range float"):
        strict_json_dumps({"value": float("nan")})

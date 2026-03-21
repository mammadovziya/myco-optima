"""Safe CSV and strict JSON serialization for downloadable analysis results."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@")


def neutralise_spreadsheet_formula(value: Any) -> Any:
    """Prefix formula-like text so spreadsheet software treats it as literal data."""

    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip(" \t\r\n")
    if value[0] in {"\t", "\r", "\n"} or stripped.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def dataframe_to_safe_csv(frame: pd.DataFrame) -> bytes:
    """Serialize a copy of a dataframe while neutralising untrusted text cells."""

    safe = frame.copy(deep=True)
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(neutralise_spreadsheet_formula)
    return safe.to_csv(index=False).encode("utf-8")


def strict_json_dumps(payload: Any, *, indent: int = 2) -> str:
    """Return standards-compliant JSON and reject NaN/Infinity instead of emitting them."""

    return json.dumps(payload, indent=indent, default=str, allow_nan=False)

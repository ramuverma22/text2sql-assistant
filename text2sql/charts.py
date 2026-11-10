"""Pick a sensible chart for a query result without asking the LLM."""
import re

import pandas as pd

_DATE_LIKE = re.compile(r"^\d{4}(-\d{2})?(-\d{2})?")


def _is_time(series: pd.Series, name: str) -> bool:
    if any(k in name.lower() for k in ("month", "date", "year", "week", "day", "period", "cohort")):
        return True
    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and sample.map(lambda v: bool(_DATE_LIKE.match(v))).all()


def choose_chart(df: pd.DataFrame) -> dict | None:
    """Return {'kind': 'line'|'bar', 'x': col, 'y': [cols]} or None for a table-only result."""
    if df is None or df.empty or len(df) < 2 or len(df.columns) < 2:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric:
        return None
    if labels:
        x = labels[0]
    else:
        # All numeric: a year/month number column can still be the x axis.
        x = next((c for c in df.columns if _is_time(df[c], c)), None)
        if x is None:
            return None
        numeric = [c for c in numeric if c != x]
        if not numeric:
            return None
    y = numeric[:3]
    if _is_time(df[x], x):
        return {"kind": "line", "x": x, "y": y}
    if len(df) <= 50:
        return {"kind": "bar", "x": x, "y": y}
    return None

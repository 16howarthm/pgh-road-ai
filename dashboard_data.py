"""Local snapshot preparation shared by the dashboard and analysis."""
import ast
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "pgh data.csv"
AGE_LABELS = ["1–30 days", "31–90 days", "91–365 days", "366+ days"]
AGE_COLORS = [[0, 105, 190], [218, 95, 0], [190, 0, 65], [100, 25, 155]]


def clean(df, today=None):
    required = {"permit_id", "active", "from_date", "to_date", "geometry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    d = df.copy()
    for c in ["from_date", "to_date"]:
        d[c + "_parsed"] = pd.to_datetime(d[c], errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    now = pd.Timestamp(today or datetime.now(ZoneInfo("America/New_York")).date()).normalize()
    d["days_past_due"] = (now - d.to_date_parsed.dt.normalize()).dt.days
    d["active_norm"] = d.active.astype("string").str.strip().str.lower().fillna("null").replace("", "null")
    d["eligible_active"] = d.active_norm.isin(["t", "true", "null"])
    d["past_due_schedule"] = d.eligible_active & d.to_date_parsed.notna() & d.days_past_due.gt(0)
    d["age_band"] = pd.cut(d.days_past_due, [0, 30, 90, 365, float("inf")], labels=AGE_LABELS)
    return d


def permit_level(d):
    # Rows without an identifier cannot be reliably counted as distinct permits.
    valid = d[d.permit_id.notna() & d.permit_id.astype(str).str.strip().ne("")]
    return valid.sort_values("days_past_due", ascending=False, kind="stable").drop_duplicates("permit_id")


def parse_path(value):
    try:
        points = ast.literal_eval(value) if isinstance(value, str) else value
        path = [[float(p[1]), float(p[0])] for p in points]
        if len(path) < 2 or any(not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in path):
            return None
        return path
    except (ValueError, TypeError, SyntaxError, IndexError):
        return None

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_ohlcv_csv(source: str | Path | IO[bytes]) -> pd.DataFrame:
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = _find_date_column(frame)
    if date_column:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column]).set_index(date_column)
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    renamed = {_match_column(frame, name): name for name in REQUIRED_COLUMNS if _match_column(frame, name)}
    frame = frame.rename(columns=renamed)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    cleaned = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    return cleaned


def _find_date_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if str(column).strip().lower() in {"date", "datetime", "timestamp", "time"}:
            return column
    return None


def _match_column(frame: pd.DataFrame, expected: str) -> str | None:
    expected_lower = expected.lower()
    for column in frame.columns:
        if str(column).strip().lower() == expected_lower:
            return column
    return None

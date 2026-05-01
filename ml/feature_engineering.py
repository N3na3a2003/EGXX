from __future__ import annotations

import pandas as pd
import numpy as np

from indicators.engine import enrich_indicators


FEATURE_COLUMNS = [
    "RSI",
    "MA50_DISTANCE",
    "MA200_DISTANCE",
    "RETURN_1D",
    "RETURN_5D",
    "RETURN_20D",
    "VOLUME_RATIO_20D",
    "VOLATILITY_20D",
]


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    data = enrich_indicators(frame)
    features = pd.DataFrame(index=data.index)
    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce").fillna(0)

    features["RSI"] = pd.to_numeric(data.get("RSI"), errors="coerce")
    features["MA50_DISTANCE"] = close / pd.to_numeric(data.get("MA50"), errors="coerce") - 1
    features["MA200_DISTANCE"] = close / pd.to_numeric(data.get("MA200"), errors="coerce") - 1
    features["RETURN_1D"] = close.pct_change(1)
    features["RETURN_5D"] = close.pct_change(5)
    features["RETURN_20D"] = close.pct_change(20)
    volume_baseline = volume.rolling(20, min_periods=10).mean()
    features["VOLUME_RATIO_20D"] = volume / volume_baseline.replace(0, np.nan)
    features["VOLATILITY_20D"] = features["RETURN_1D"].rolling(20, min_periods=10).std()
    return features.replace([float("inf"), float("-inf")], np.nan)


def build_training_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = build_feature_frame(frame)
    if frame.empty:
        return features, pd.Series(dtype=int)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    target = (close.shift(-1) > close).astype(int)
    dataset = features.copy()
    dataset["TARGET"] = target
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["TARGET"])
    if dataset.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)
    return dataset[FEATURE_COLUMNS], dataset["TARGET"].astype(int)


def latest_feature_row(frame: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(frame)
    if features.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    clean = features.dropna(subset=FEATURE_COLUMNS)
    if clean.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return clean.tail(1)

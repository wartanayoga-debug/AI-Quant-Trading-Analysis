from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "symbol", "market", "timeframe", "open", "high", "low", "close", "volume"]


@dataclass
class DataQualityResult:
    ok: bool
    status: str
    cleaned: pd.DataFrame
    issues: List[str]
    dropped_rows: int
    min_required: int


def minimum_candles(timeframe: str, daily_min: int = 250, intraday_min: int = 500) -> int:
    return daily_min if str(timeframe).lower() in {"1d", "1day", "d"} else intraday_min


def validate_ohlcv(
    df: pd.DataFrame,
    min_candles: int | None = None,
    daily_min: int = 250,
    intraday_min: int = 500,
) -> DataQualityResult:
    issues: List[str] = []
    if df is None or df.empty:
        return DataQualityResult(False, "empty_dataset", pd.DataFrame(), ["dataset_empty"], 0, min_candles or intraday_min)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return DataQualityResult(False, "missing_columns", pd.DataFrame(), [f"missing:{','.join(missing)}"], 0, min_candles or intraday_min)

    original_len = len(df)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out = out[out["volume"] >= 0]
    out = out.sort_values(["symbol", "market", "timeframe", "timestamp"])
    before_dedup = len(out)
    out = out.drop_duplicates(["symbol", "market", "timeframe", "timestamp"], keep="last")
    if before_dedup != len(out):
        issues.append("duplicate_candles_removed")

    if not out.empty:
        bad_high_low = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        if bad_high_low.any():
            issues.append("bad_high_low_rows_removed")
            out = out.loc[~bad_high_low].copy()

    if len(out) != original_len:
        issues.append("invalid_rows_removed")

    if min_candles is None:
        tf = str(out["timeframe"].iloc[-1]) if not out.empty else ""
        min_candles = minimum_candles(tf, daily_min, intraday_min)

    if len(out) < min_candles:
        issues.append(f"insufficient_data:{len(out)}/{min_candles}")
        return DataQualityResult(False, "insufficient_data", out.reset_index(drop=True), issues, original_len - len(out), min_candles)

    monotonic_ok = out.groupby(["symbol", "market", "timeframe"])["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()
    if not bool(monotonic_ok):
        issues.append("timestamp_not_sorted")

    return DataQualityResult(True, "ok", out.reset_index(drop=True), issues, original_len - len(out), min_candles)


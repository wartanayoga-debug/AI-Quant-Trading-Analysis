from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {
    "timestamp",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target_buy_valid",
    "target_sell_valid",
    "tp_before_sl",
    "direction_5",
    "direction_10",
    "label_rr",
}


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _features_one_group(group: pd.DataFrame, group_key: tuple | None = None) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    if isinstance(group_key, tuple) and len(group_key) == 3:
        symbol, market, timeframe = group_key
        if "symbol" not in g.columns:
            g["symbol"] = symbol
        if "market" not in g.columns:
            g["market"] = market
        if "timeframe" not in g.columns:
            g["timeframe"] = timeframe
    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]

    for span in [9, 20, 50, 100, 200]:
        g[f"ema_{span}"] = _ema(close, span)

    g["close_above_ema20"] = (close > g["ema_20"]).astype(int)
    g["close_above_ema50"] = (close > g["ema_50"]).astype(int)
    g["ema20_above_ema50"] = (g["ema_20"] > g["ema_50"]).astype(int)
    g["ema50_above_ema200"] = (g["ema_50"] > g["ema_200"]).astype(int)
    g["trend_strength_score"] = (
        g[["close_above_ema20", "close_above_ema50", "ema20_above_ema50", "ema50_above_ema200"]].sum(axis=1) / 4
    )

    g["rsi_14"] = _rsi(close, 14)
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    g["macd"] = ema_12 - ema_26
    g["macd_signal"] = _ema(g["macd"], 9)
    g["macd_hist"] = g["macd"] - g["macd_signal"]
    lowest_14 = low.rolling(14, min_periods=14).min()
    highest_14 = high.rolling(14, min_periods=14).max()
    g["stochastic_k"] = 100 * (close - lowest_14) / (highest_14 - lowest_14).replace(0, np.nan)
    g["stochastic_d"] = g["stochastic_k"].rolling(3, min_periods=3).mean()
    g["roc_10"] = close.pct_change(10)

    g["atr_14"] = _atr(g, 14)
    g["atr_percent"] = g["atr_14"] / close.replace(0, np.nan)
    g["bollinger_middle"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    g["bollinger_upper"] = g["bollinger_middle"] + 2 * bb_std
    g["bollinger_lower"] = g["bollinger_middle"] - 2 * bb_std
    g["bollinger_width"] = (g["bollinger_upper"] - g["bollinger_lower"]) / g["bollinger_middle"].replace(0, np.nan)
    g["close_position_in_bollinger"] = (close - g["bollinger_lower"]) / (g["bollinger_upper"] - g["bollinger_lower"]).replace(0, np.nan)

    g["volume_sma_20"] = volume.rolling(20, min_periods=20).mean()
    g["volume_ratio"] = volume / g["volume_sma_20"].replace(0, np.nan)
    g["volume_spike"] = (g["volume_ratio"] >= 1.5).astype(int)
    g["obv"] = _obv(close, volume)

    candle_range = (high - low).replace(0, np.nan)
    body = (close - g["open"]).abs()
    g["candle_body_percent"] = body / close.replace(0, np.nan)
    g["upper_wick_percent"] = (high - np.maximum(g["open"], close)) / candle_range
    g["lower_wick_percent"] = (np.minimum(g["open"], close) - low) / candle_range
    g["is_bullish_candle"] = (close > g["open"]).astype(int)
    g["is_bearish_candle"] = (close < g["open"]).astype(int)
    g["body_to_range_ratio"] = body / candle_range

    g["rolling_high_20"] = high.rolling(20, min_periods=20).max()
    g["rolling_low_20"] = low.rolling(20, min_periods=20).min()
    g["distance_to_high_20"] = (g["rolling_high_20"] - close) / close.replace(0, np.nan)
    g["distance_to_low_20"] = (close - g["rolling_low_20"]) / close.replace(0, np.nan)
    g["breakout_20"] = (close > g["rolling_high_20"].shift(1)).astype(int)
    g["breakdown_20"] = (close < g["rolling_low_20"].shift(1)).astype(int)

    for window in [1, 3, 5]:
        g[f"return_{window}"] = close.pct_change(window)
    g["volatility_10"] = g["return_1"].rolling(10, min_periods=10).std()
    g["volatility_20"] = g["return_1"].rolling(20, min_periods=20).std()

    g["market_type"] = g["market"].astype(str).str.lower()
    g["timeframe_feature"] = g["timeframe"].astype(str).str.lower()
    ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
    g["hour_of_day"] = ts.dt.hour.fillna(0).astype(int)
    g["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
    
    market_str = g["market"].iloc[0] if len(g) > 0 else ""
    is_idx = "idx" in str(market_str).lower()
    
    # Session features (IDX specific)
    if is_idx:
        prev_close = close.shift(1)
        g["overnight_gap_pct"] = (g["open"] - prev_close) / prev_close.replace(0, np.nan)
        minute_of_hour = ts.dt.minute.fillna(0).astype(int)
        g["idx_morning_session"] = ((g["hour_of_day"] == 9) | (g["hour_of_day"] == 10) | ((g["hour_of_day"] == 11) & (minute_of_hour <= 30))).astype(int)
    else:
        g["overnight_gap_pct"] = 0.0
        g["idx_morning_session"] = 0
        
    # Liquidity features
    g["dollar_volume"] = close * volume
    g["dollar_volume_sma_20"] = g["dollar_volume"].rolling(20, min_periods=20).mean()
    ret_abs = g["return_1"].abs()
    g["amihud_illiq"] = ret_abs / g["dollar_volume"].replace(0, np.nan)
    g["amihud_illiq_20"] = g["amihud_illiq"].rolling(20, min_periods=20).mean()
    
    g["kyle_lambda"] = (high - low) / volume.replace(0, np.nan)
    g["kyle_lambda_20"] = g["kyle_lambda"].rolling(20, min_periods=20).mean()
    
    vol_std = volume.rolling(20, min_periods=20).std()
    g["volume_zscore_20"] = (volume - g["volume_sma_20"]) / vol_std.replace(0, np.nan)
    
    return g


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    required = {"timestamp", "symbol", "market", "timeframe", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    sorted_df = df.copy().sort_values(["symbol", "market", "timeframe", "timestamp"])
    parts = [
        _features_one_group(group, group_key)
        for group_key, group in sorted_df.groupby(["symbol", "market", "timeframe"], group_keys=False)
    ]
    out = pd.concat(parts, ignore_index=True) if parts else sorted_df
    out = out.replace([np.inf, -np.inf], np.nan)
    if dropna:
        feature_cols = [col for col in out.columns if col not in NON_FEATURE_COLUMNS]
        out = out.dropna(subset=[col for col in feature_cols if col not in {"symbol", "market", "timeframe", "market_type", "timeframe_feature"}])
    return out.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]

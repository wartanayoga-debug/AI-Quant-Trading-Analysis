from __future__ import annotations

import numpy as np
import pandas as pd


def _first_touch_label(
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    horizon: int,
    tp: float,
    sl: float,
    side: str,
) -> float:
    end = min(len(highs), start_idx + horizon + 1)
    if end <= start_idx + horizon:
        return np.nan
    for j in range(start_idx + 1, end):
        if side == "buy":
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl
        if tp_hit and sl_hit:
            return 0.0
        if tp_hit:
            return 1.0
        if sl_hit:
            return 0.0
    return 0.0


def add_tp_sl_labels(
    df: pd.DataFrame,
    horizon: int = 20,
    tp_atr_multiplier: float = 2.0,
    sl_atr_multiplier: float = 1.0,
    min_rr: float = 1.5,
) -> pd.DataFrame:
    required = {"symbol", "market", "timeframe", "high", "low", "close", "atr_14"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns for labels: {sorted(missing)}")

    frames = []
    for _, group in df.sort_values("timestamp").groupby(["symbol", "market", "timeframe"], group_keys=False):
        g = group.copy()
        highs = g["high"].to_numpy(dtype=float)
        lows = g["low"].to_numpy(dtype=float)
        closes = g["close"].to_numpy(dtype=float)
        atr = g["atr_14"].to_numpy(dtype=float)
        buy_labels = []
        sell_labels = []
        rr_values = []
        for i, close in enumerate(closes):
            if not np.isfinite(close) or not np.isfinite(atr[i]) or atr[i] <= 0:
                buy_labels.append(np.nan)
                sell_labels.append(np.nan)
                rr_values.append(np.nan)
                continue
            risk = sl_atr_multiplier * atr[i]
            reward = tp_atr_multiplier * atr[i]
            rr = reward / max(1e-12, risk)
            rr_values.append(rr)
            if rr < min_rr:
                buy_labels.append(0)
                sell_labels.append(0)
                continue
            buy_labels.append(_first_touch_label(highs, lows, i, horizon, close + reward, close - risk, "buy"))
            sell_labels.append(_first_touch_label(highs, lows, i, horizon, close - reward, close + risk, "sell"))
        g["target_buy_valid"] = buy_labels
        g["target_sell_valid"] = sell_labels
        g["tp_before_sl"] = np.maximum(g["target_buy_valid"], g["target_sell_valid"])
        for lookahead in (5, 10):
            future_close = g["close"].shift(-lookahead)
            g[f"direction_{lookahead}"] = np.where(
                future_close.notna(),
                (future_close > g["close"]).astype(int),
                np.nan,
            )
        g["label_rr"] = rr_values
        frames.append(g)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.dropna(subset=["target_buy_valid", "target_sell_valid"])
    out["target_buy_valid"] = out["target_buy_valid"].astype(int)
    out["target_sell_valid"] = out["target_sell_valid"].astype(int)
    out["tp_before_sl"] = out["tp_before_sl"].astype(int)
    
    # Do NOT drop direction_5 and direction_10 here because they are allowed targets.
    # feature_engineering.py's NON_FEATURE_COLUMNS already prevents them from leaking into inputs.
    
    return out.reset_index(drop=True)

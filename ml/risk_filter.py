from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import load_config


@dataclass
class RiskFilterResult:
    passed: bool
    signal: str
    entry_reference: float | None
    stop_loss: float | None
    take_profit: float | None
    risk_reward_ratio: float
    reasons: List[str]


def build_trade_plan(row: pd.Series, side: str = "BUY") -> tuple[float | None, float | None, float | None, float]:
    close = float(row.get("close", np.nan))
    atr = float(row.get("atr_14", np.nan))
    if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
        return None, None, None, 0.0
    cfg = load_config()
    risk = cfg.sl_atr_multiplier * atr
    reward = cfg.tp_atr_multiplier * atr
    if side == "SELL":
        entry = close
        sl = close + risk
        tp = close - reward
    else:
        entry = close
        sl = close - risk
        tp = close + reward
    rr = reward / max(1e-12, risk)
    return entry, sl, tp, float(rr)


def apply_risk_filter(row: pd.Series, probability: float, side: str = "BUY") -> RiskFilterResult:
    cfg = load_config()
    reasons: List[str] = []
    entry, sl, tp, rr = build_trade_plan(row, side)
    threshold = cfg.confidence_threshold_sell if side == "SELL" else cfg.confidence_threshold_buy
    atr_percent = float(row.get("atr_percent", np.nan))
    volume_ratio = float(row.get("volume_ratio", 0))
    close = float(row.get("close", np.nan))
    ema20 = float(row.get("ema_20", np.nan))
    ema50 = float(row.get("ema_50", np.nan))
    market = str(row.get("market", "")).lower()
    max_atr = cfg.max_atr_percent_crypto if "crypto" in market else cfg.max_atr_percent_idx

    if probability < threshold:
        reasons.append(f"probability {probability:.3f} below threshold {threshold:.2f}")
    if entry is None or sl is None or tp is None:
        reasons.append("invalid ATR trade plan")
    if rr < cfg.min_risk_reward:
        reasons.append(f"risk_reward_ratio {rr:.2f} below {cfg.min_risk_reward:.2f}")
    if not np.isfinite(volume_ratio) or volume_ratio < cfg.min_volume_ratio:
        reasons.append(f"volume_ratio {volume_ratio:.2f} below {cfg.min_volume_ratio:.2f}")
    if not np.isfinite(atr_percent) or atr_percent <= 0:
        reasons.append("atr_percent unavailable")
    elif atr_percent > max_atr:
        reasons.append(f"atr_percent {atr_percent:.3f} above max {max_atr:.3f}")
    if np.isfinite(close) and np.isfinite(ema20) and np.isfinite(ema50):
        distance = min(abs(close - ema20), abs(close - ema50)) / max(1e-12, close)
        if distance > 0.08:
            reasons.append("price too far from EMA20/EMA50")

    passed = len(reasons) == 0
    signal = side if passed else "WAIT"
    if side == "SELL" and not passed and ("ema50_above_ema200" in row and int(row.get("ema50_above_ema200", 1)) == 0):
        signal = "EXIT WARNING"
    return RiskFilterResult(passed, signal, entry, sl, tp, rr, reasons)


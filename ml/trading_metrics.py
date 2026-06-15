from __future__ import annotations

import numpy as np
import pandas as pd


def profit_factor(r_values: np.ndarray) -> float:
    wins = r_values[r_values > 0].sum()
    losses = abs(r_values[r_values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown_r(r_values: np.ndarray) -> float:
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)

    return float(ece)


def evaluate_trading_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    prob_col: str,
    r_col: str,
    threshold: float = 0.75,
) -> dict:
    if y_true_col not in df.columns:
        raise ValueError(f"Missing y_true_col: {y_true_col}")
    if prob_col not in df.columns:
        raise ValueError(f"Missing prob_col: {prob_col}")
    if r_col not in df.columns:
        raise ValueError(f"Missing r_col: {r_col}")

    subset = df[df[prob_col] >= threshold].copy()

    if subset.empty:
        return {
            "trade_count_at_threshold": 0,
            "average_r_at_threshold": 0.0,
            "profit_factor_at_threshold": 0.0,
            "max_drawdown_r_at_threshold": 0.0,
            "precision_at_threshold": 0.0,
            "expectancy_r_at_threshold": 0.0,
            "ece": expected_calibration_error(df[y_true_col].values, df[prob_col].values),
        }

    r = subset[r_col].astype(float).values
    y = subset[y_true_col].astype(int).values

    return {
        "trade_count_at_threshold": int(len(subset)),
        "average_r_at_threshold": float(np.mean(r)),
        "profit_factor_at_threshold": profit_factor(r),
        "max_drawdown_r_at_threshold": max_drawdown_r(r),
        "precision_at_threshold": float(np.mean(y)),
        "expectancy_r_at_threshold": float(np.mean(r)),
        "ece": expected_calibration_error(df[y_true_col].values, df[prob_col].values),
    }

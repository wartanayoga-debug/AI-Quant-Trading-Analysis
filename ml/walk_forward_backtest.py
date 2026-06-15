from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score

from .feature_engineering import get_feature_columns
from .feature_guard import assert_no_leaky_features
from .train_autogluon import ALLOWED_TARGETS, gpu_sanity_check, prepare_training_frame
from .config import load_config

def iter_purged_walk_forward_folds(
    data: pd.DataFrame,
    folds: int,
    horizon_candles: int,
    embargo_candles: int,
    min_train_rows: int = 200,
    min_test_rows: int = 50,
):
    data = data.sort_values("timestamp").reset_index(drop=True)
    n = len(data)

    if folds < 1:
        raise ValueError("folds must be >= 1")

    fold_size = max(min_test_rows, n // (folds + 2))
    purge = max(1, horizon_candles)
    embargo = max(0, embargo_candles)

    for fold in range(folds):
        train_end = fold_size * (fold + 2)
        train_purged_end = train_end - purge
        test_start = train_end + embargo
        test_end = min(n, test_start + fold_size)

        if train_purged_end < min_train_rows:
            continue

        if test_end - test_start < min_test_rows:
            continue

        train_df = data.iloc[:train_purged_end].copy()
        test_df = data.iloc[test_start:test_end].copy()

        yield fold, train_df, test_df

def summarize_signals(df: pd.DataFrame, prob_col: str, target: str, threshold: float) -> Dict[str, Any]:
    signals = df[df[prob_col] >= threshold].copy()
    wins = int(signals[target].sum()) if not signals.empty else 0
    losses = int(len(signals) - wins)
    returns = signals[target].map({1: 1.0, 0: -1.0}).to_numpy(dtype=float) if not signals.empty else np.array([])
    equity = returns.cumsum() if len(returns) else np.array([0.0])
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return {
        "total_signals": int(len(signals)),
        "valid_signals": wins,
        "invalid_signals": losses,
        "winrate": float(wins / len(signals)) if len(signals) else 0.0,
        "profit_factor": float(wins / max(1, losses)),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "average_return_per_signal": float(returns.mean()) if len(returns) else 0.0,
        "median_return_per_signal": float(np.median(returns)) if len(returns) else 0.0,
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
    }


def walk_forward_backtest(csv_path: str | Path, target: str = "target_buy_valid", folds: int = 3) -> Dict[str, Any]:
    cfg = load_config()
    data = prepare_training_frame(csv_path, target).sort_values("timestamp").reset_index(drop=True)
    if len(data) < cfg.min_train_rows:
        raise ValueError(f"not enough rows for walk-forward: {len(data)}")
    gpu = gpu_sanity_check()
    use_gpu = gpu.get("cuda_available") and cfg.num_gpus > 0
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise RuntimeError("AutoGluon is not installed. Install autogluon before walk-forward backtest.") from exc

    feature_cols = get_feature_columns(data)
    feature_cols = assert_no_leaky_features(feature_cols, label=target)
    
    fold_results: List[Dict[str, Any]] = []
    
    for fold, train_df, test_df in iter_purged_walk_forward_folds(
        data=data,
        folds=folds,
        horizon_candles=cfg.horizon_candles,
        embargo_candles=max(cfg.embargo_candles, cfg.horizon_candles),
        min_train_rows=cfg.min_train_rows,
        min_test_rows=cfg.min_valid_rows,
    ):
        # Use last 20% of training data as validation to avoid leaking test data
        val_split = max(1, int(len(train_df) * 0.8))
        train_split_df = train_df.iloc[:val_split]
        val_split_df = train_df.iloc[val_split:]
        
        if train_split_df[target].nunique(dropna=True) < 2:
            fold_results.append({
                "fold": fold + 1,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "skipped": True,
                "reason": "single_class_training_split",
            })
            continue
        model_path = Path(cfg.backtest_dir) / f"fold_{fold + 1}_predictor"
        predictor = TabularPredictor(label=target, problem_type=cfg.problem_type, eval_metric=cfg.eval_metric, path=str(model_path)).fit(
            train_data=train_split_df[feature_cols + [target]],
            tuning_data=val_split_df[feature_cols + [target]],
            presets=cfg.presets if use_gpu else cfg.cpu_fallback_presets,
            time_limit=min(cfg.time_limit, 1800),
            num_cpus=cfg.num_cpus,
            num_gpus=cfg.num_gpus if use_gpu else 0,
            use_bag_holdout=True,
        )
        proba = predictor.predict_proba(test_df[feature_cols])
        y_prob = proba[1] if 1 in proba.columns else proba["1"]
        scored = test_df.copy()
        scored["probability"] = y_prob.astype(float)
        fold_results.append({
            "fold": fold + 1,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "roc_auc": float(roc_auc_score(test_df[target], y_prob)) if len(set(test_df[target])) > 1 else 0.0,
            "precision_at_0_75": float(precision_score(test_df[target], scored["probability"] >= 0.75, zero_division=0)),
            "summary": summarize_signals(scored, "probability", target, 0.75),
            "performance_by_symbol": scored.groupby("symbol").apply(lambda g: summarize_signals(g, "probability", target, 0.75)).to_dict(),
            "performance_by_timeframe": scored.groupby("timeframe").apply(lambda g: summarize_signals(g, "probability", target, 0.75)).to_dict(),
            "performance_by_market": scored.groupby("market").apply(lambda g: summarize_signals(g, "probability", target, 0.75)).to_dict(),
        })
    result = {"ok": True, "folds": fold_results}
    out_path = Path(cfg.backtest_dir) / "walk_forward_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest AutoGluon classifier")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="target_buy_valid", choices=ALLOWED_TARGETS)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(walk_forward_backtest(args.csv, args.target, args.folds), indent=2))


if __name__ == "__main__":
    main()

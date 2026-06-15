from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import load_config
from .data_quality import validate_ohlcv
from .feature_engineering import build_features, get_feature_columns
from .label_builder import add_tp_sl_labels
from .logging_utils import get_logger
from .model_registry import compare_model_with_current, register_model
from .time_split import PurgedSplitConfig, purged_embargo_split
from .feature_guard import assert_no_leaky_features
from .trading_metrics import evaluate_trading_metrics
from .promotion_gate import PromotionGateConfig, evaluate_promotion_gate


logger = get_logger("autogluon_training", "training.log")
ALLOWED_TARGETS = ("target_buy_valid", "target_sell_valid", "tp_before_sl", "direction_5", "direction_10")


def gpu_sanity_check() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        return {"torch_available": False, "cuda_available": False, "cuda_device_count": 0, "error": str(exc)}


def _metrics(y_true, y_prob, threshold: float = 0.75) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    y_thr = (y_prob >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "signals_at_0_75": int(y_thr.sum()),
        "precision_at_0_75": float(precision_score(y_true, y_thr, zero_division=0)),
    }
    for thr in [0.70, 0.75, 0.80]:
        pred = (y_prob >= thr).astype(int)
        out[f"precision_at_{str(thr).replace('.', '_')}"] = float(precision_score(y_true, pred, zero_division=0))
        out[f"signals_at_{str(thr).replace('.', '_')}"] = int(pred.sum())
    return out


def prepare_training_frame(csv_path: str | Path, target: str = "target_buy_valid") -> pd.DataFrame:
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"unsupported target: {target}. allowed targets: {', '.join(ALLOWED_TARGETS)}")
    cfg = load_config()
    source_path = Path(csv_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {source_path}. "
            "Generate it first with: npm run export:autogluon-data -- --out data/training/ohlcv.csv"
        )
    raw = pd.read_csv(source_path)
    quality = validate_ohlcv(raw, min_candles=cfg.min_symbol_candles_daily)
    if quality.cleaned.empty:
        raise ValueError(f"dataset is not usable: {quality.status} {quality.issues}")
    features = build_features(quality.cleaned, dropna=True)
    labeled = add_tp_sl_labels(
        features,
        horizon=cfg.horizon_candles,
        tp_atr_multiplier=cfg.tp_atr_multiplier,
        sl_atr_multiplier=cfg.sl_atr_multiplier,
        min_rr=cfg.min_risk_reward,
    )
    if target not in labeled.columns:
        raise ValueError(f"target column missing: {target}")
    labeled = labeled.dropna(subset=[target]).copy()
    labeled[target] = labeled[target].astype(int)
    invalid_values = sorted(set(labeled[target].dropna().unique()).difference({0, 1}))
    if invalid_values:
        raise ValueError(f"target {target} must be binary 0/1; got {invalid_values}")
    return labeled.reset_index(drop=True)


def train_from_csv(csv_path: str | Path, target: str = "target_buy_valid") -> Dict[str, Any]:
    cfg = load_config()
    logger.info("training start csv=%s target=%s", csv_path, target)
    gpu = gpu_sanity_check()
    df = prepare_training_frame(csv_path, target)
    if len(df) < cfg.min_train_rows:
        raise ValueError(f"training rows {len(df)} below min_train_rows {cfg.min_train_rows}")

    split_cfg = PurgedSplitConfig(
        train_ratio=getattr(cfg, "train_ratio", 0.8),
        horizon_candles=getattr(cfg, "horizon_candles", 20),
        embargo_candles=max(
            getattr(cfg, "embargo_candles", 20),
            getattr(cfg, "horizon_candles", 20),
        ),
        min_train_rows=getattr(cfg, "min_train_rows", 200),
        min_valid_rows=getattr(cfg, "min_valid_rows", 50),
    )

    train_df, valid_df = purged_embargo_split(df, split_cfg)
    logger.info(
        "Purged split complete: train=%s, valid=%s, horizon=%s, embargo=%s",
        len(train_df),
        len(valid_df),
        split_cfg.horizon_candles,
        split_cfg.embargo_candles,
    )

    feature_cols = get_feature_columns(train_df)
    feature_cols = assert_no_leaky_features(feature_cols, label=target)
    
    # We keep label_rr in the datasets for trading metric calculations
    train_data = train_df[feature_cols + [target, "label_rr"]]
    valid_data = valid_df[feature_cols + [target, "label_rr"]]
    target_distribution = train_data[target].value_counts(dropna=False).to_dict()
    if train_data[target].nunique(dropna=True) < 2:
        raise ValueError(f"target {target} has only one class in training split: {target_distribution}")
    minority_rows = int(train_data[target].value_counts(dropna=False).min())
    if minority_rows < 25:
        raise ValueError(f"target {target} minority class has only {minority_rows} rows; collect more balanced data before training")
    logger.info("dataset rows=%d train=%d valid=%d target_distribution=%s", len(df), len(train_data), len(valid_data), target_distribution)

    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise RuntimeError("AutoGluon is not installed. Install autogluon before training.") from exc

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = Path(cfg.model_dir) / "versions" / version / "predictor"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    use_gpu = gpu.get("cuda_available") and cfg.num_gpus > 0
    fit_kwargs = {
        "train_data": train_data,
        "tuning_data": valid_data,
        "presets": cfg.presets if use_gpu else cfg.cpu_fallback_presets,
        "time_limit": cfg.time_limit,
        "num_cpus": cfg.num_cpus,
        "num_gpus": cfg.num_gpus if use_gpu else 0,
        "use_bag_holdout": True,
    }
    
    eval_metric = cfg.eval_metric

    if getattr(cfg, "enable_custom_trading_metric", False):
        from ml.custom_metrics import build_optional_profit_scorer

        scorer = build_optional_profit_scorer()
        if scorer is not None:
            eval_metric = scorer
        else:
            logger.warning("Custom trading metric unavailable; falling back to %s", cfg.eval_metric)
            
    predictor = TabularPredictor(label=target, problem_type=cfg.problem_type, eval_metric=eval_metric, path=str(model_path)).fit(**fit_kwargs)
    proba = predictor.predict_proba(valid_data[feature_cols])
    y_prob = proba[1] if 1 in proba.columns else proba["1"]
    
    valid_data_scored = valid_data.copy()
    valid_data_scored["probability"] = y_prob
    valid_data_scored["r_value"] = np.where(valid_data_scored[target] == 1, valid_data_scored["label_rr"], -1.0)
    
    metrics = _metrics(valid_data[target].astype(int), y_prob.astype(float))
    
    trading_metrics = evaluate_trading_metrics(
        df=valid_data_scored,
        y_true_col=target,
        prob_col="probability",
        r_col="r_value",
        threshold=0.75,
    )
    metrics.update(trading_metrics)
    
    metrics.update({
        "target": target,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_data)),
        "rows_valid": int(len(valid_data)),
        "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
    })
    
    promotion_cfg = PromotionGateConfig()
    promotion_report = evaluate_promotion_gate(metrics, promotion_cfg)
    
    leaderboard = predictor.leaderboard(valid_data, silent=True)
    version_dir = model_path.parent
    leaderboard.to_csv(version_dir / "leaderboard.csv", index=False)
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (version_dir / "training_config.json").write_text(json.dumps(cfg.as_dict(), indent=2), encoding="utf-8")
    (version_dir / "gpu_sanity.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")
    (version_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (version_dir / "promotion_report.json").write_text(json.dumps(promotion_report, indent=2), encoding="utf-8")
    
    current_model = get_model_metadata("latest")
    is_first_model = not current_model.get("available")
    passed_gate = promotion_report.get("passed", False)
    
    if passed_gate:
        if is_first_model:
            alias = "latest"
            status = "PROMOTED"
        else:
            alias = "challenger"
            status = "CHALLENGER_MODEL"
    else:
        alias = None
        status = "CANDIDATE_REJECTED"

    metadata = register_model(version, metrics, model_path, alias=alias)
    metadata["promotion_report"] = promotion_report
    metadata["status"] = status
    
    logger.info("training complete version=%s status=%s metrics=%s", version, status, metrics)
    return {"ok": True, "version": version, "promoted": (alias == "latest"), "challenger": (alias == "challenger"), "metadata": metadata, "metrics": metrics, "gpu": gpu}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AutoGluon signal classifier")
    parser.add_argument("--csv", required=True, help="OHLCV CSV path")
    parser.add_argument("--target", default="target_buy_valid", choices=ALLOWED_TARGETS)
    args = parser.parse_args()
    print(json.dumps(train_from_csv(args.csv, args.target), indent=2))


if __name__ == "__main__":
    main()

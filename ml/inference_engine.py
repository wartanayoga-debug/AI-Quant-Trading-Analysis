from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .config import load_config
from .data_quality import validate_ohlcv
from .feature_engineering import build_features, get_feature_columns
from .logging_utils import get_logger
from .model_registry import get_model_metadata, load_model
from .risk_filter import apply_risk_filter


logger = get_logger("autogluon_inference", "inference.log")


def _positive_probability(predictor: Any, row: pd.DataFrame) -> float:
    proba = predictor.predict_proba(row)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            return float(proba[1].iloc[0])
        if "1" in proba.columns:
            return float(proba["1"].iloc[0])
        if True in proba.columns:
            return float(proba[True].iloc[0])
        raise RuntimeError("positive class probability not found in predict_proba output")
    raise RuntimeError("predict_proba returned unsupported output")


def screen_ohlcv(candles: List[Dict[str, Any]], side: str = "BUY") -> Dict[str, Any]:
    cfg = load_config()
    df = pd.DataFrame(candles)
    quality = validate_ohlcv(df, min_candles=None, daily_min=cfg.min_symbol_candles_daily, intraday_min=cfg.min_symbol_candles_intraday)
    if not quality.ok:
        return {"signal": "WAIT", "confidence": 0.0, "data_status": quality.status, "reasons": quality.issues}

    metadata = get_model_metadata("latest")
    if not metadata.get("available"):
        return {"signal": "WAIT", "confidence": 0.0, "data_status": "model_unavailable", "reasons": ["AutoGluon latest model is not available."]}

    try:
        features = build_features(quality.cleaned)
        if features.empty:
            return {"signal": "WAIT", "confidence": 0.0, "data_status": "insufficient_feature_rows", "reasons": ["Feature rows are empty after rolling indicators."]}
        latest = features.tail(1)
        feature_cols = get_feature_columns(latest)
        predictor = load_model("latest")
        probability = _positive_probability(predictor, latest[feature_cols])
        side = side.upper()
        risk = apply_risk_filter(latest.iloc[0], probability, "SELL" if side == "SELL" else "BUY")
        row = latest.iloc[0]
        signal = risk.signal
        if signal == "BUY" and probability >= cfg.confidence_threshold_strong_buy and risk.risk_reward_ratio >= cfg.min_risk_reward * 1.25:
            signal = "STRONG BUY"
        technical_reasons = [
            f"trend_strength_score={float(row.get('trend_strength_score', 0)):.2f}",
            f"volume_ratio={float(row.get('volume_ratio', 0)):.2f}",
            f"rsi_14={float(row.get('rsi_14', 0)):.1f}",
            f"macd_hist={float(row.get('macd_hist', 0)):.6f}",
        ]
        if risk.reasons:
            technical_reasons.extend(risk.reasons)
        try:
            challenger_meta = get_model_metadata("challenger")
            challenger_payload = None
            if challenger_meta.get("available"):
                c_predictor = load_model("challenger")
                c_prob = _positive_probability(c_predictor, latest[feature_cols])
                c_risk = apply_risk_filter(latest.iloc[0], c_prob, "SELL" if side == "SELL" else "BUY")
                c_signal = c_risk.signal
                if c_signal == "BUY" and c_prob >= cfg.confidence_threshold_strong_buy and c_risk.risk_reward_ratio >= cfg.min_risk_reward * 1.25:
                    c_signal = "STRONG BUY"
                challenger_payload = {
                    "version": challenger_meta.get("version"),
                    "signal": c_signal,
                    "confidence": round(c_prob, 4),
                    "entry_reference": c_risk.entry_reference,
                    "stop_loss": c_risk.stop_loss,
                    "take_profit": c_risk.take_profit,
                    "risk_reward_ratio": round(c_risk.risk_reward_ratio, 4)
                }
        except Exception as e:
            logger.warning("Challenger inference failed: %s", e)
            challenger_payload = None

        return {
            "symbol": str(row.get("symbol")),
            "market": str(row.get("market")),
            "timeframe": str(row.get("timeframe")),
            "signal": signal,
            "confidence": round(probability, 4),
            "probability_buy_valid": round(probability, 4) if side != "SELL" else None,
            "probability_sell_valid": round(probability, 4) if side == "SELL" else None,
            "entry_reference": risk.entry_reference,
            "stop_loss": risk.stop_loss,
            "take_profit": risk.take_profit,
            "risk_reward_ratio": round(risk.risk_reward_ratio, 4),
            "technical_reasons": technical_reasons,
            "model_version": metadata.get("version"),
            "trained_at": metadata.get("registered_at"),
            "backtest_summary": metadata.get("metrics", {}),
            "last_candle_time": str(row.get("timestamp")),
            "data_status": "ok",
            "challenger": challenger_payload,
        }
    except Exception as exc:
        logger.exception("inference failed")
        return {"signal": "WAIT", "confidence": 0.0, "data_status": "inference_error", "reasons": [str(exc)]}


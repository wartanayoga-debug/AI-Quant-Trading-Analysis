from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AutoGluonConfig:
    enabled: bool = True
    problem_type: str = "binary"
    eval_metric: str = "roc_auc"
    enable_custom_trading_metric: bool = False
    presets: str = "best_quality"
    cpu_fallback_presets: str = "medium_quality"
    time_limit: int = 7200
    num_gpus: int = 1
    num_cpus: int = 20
    confidence_threshold_buy: float = 0.75
    confidence_threshold_strong_buy: float = 0.85
    confidence_threshold_sell: float = 0.75
    min_risk_reward: float = 1.5
    min_volume_ratio: float = 1.0
    max_atr_percent_crypto: float = 0.12
    max_atr_percent_idx: float = 0.08
    horizon_candles: int = 20
    embargo_candles: int = 20
    train_ratio: float = 0.8
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.0
    min_train_rows: int = 5000
    min_valid_rows: int = 50
    min_symbol_candles_daily: int = 250
    min_symbol_candles_intraday: int = 500
    max_symbols_per_batch: int = 500
    model_dir: str = str(ROOT_DIR / "models" / "autogluon")
    training_dir: str = str(ROOT_DIR / "data" / "training")
    backtest_dir: str = str(ROOT_DIR / "data" / "backtest")
    log_dir: str = str(ROOT_DIR / "logs")
    use_gpu_fallback: bool = True
    min_precision_at_threshold: float = 0.55
    min_backtest_signals: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def load_config() -> AutoGluonConfig:
    base = AutoGluonConfig()
    return AutoGluonConfig(
        enabled=_env_bool("AUTOGLUON_ENABLED", base.enabled),
        problem_type=os.getenv("AUTOGLUON_PROBLEM_TYPE", base.problem_type),
        eval_metric=os.getenv("AUTOGLUON_EVAL_METRIC", base.eval_metric),
        enable_custom_trading_metric=_env_bool("AUTOGLUON_ENABLE_CUSTOM_TRADING_METRIC", base.enable_custom_trading_metric),
        presets=os.getenv("AUTOGLUON_PRESETS", base.presets),
        cpu_fallback_presets=os.getenv("AUTOGLUON_CPU_FALLBACK_PRESETS", base.cpu_fallback_presets),
        time_limit=_env_int("AUTOGLUON_TIME_LIMIT", base.time_limit),
        num_gpus=_env_int("AUTOGLUON_NUM_GPUS", base.num_gpus),
        num_cpus=_env_int("AUTOGLUON_NUM_CPUS", base.num_cpus),
        confidence_threshold_buy=_env_float("AUTOGLUON_CONFIDENCE_BUY", base.confidence_threshold_buy),
        confidence_threshold_strong_buy=_env_float("AUTOGLUON_CONFIDENCE_STRONG_BUY", base.confidence_threshold_strong_buy),
        confidence_threshold_sell=_env_float("AUTOGLUON_CONFIDENCE_SELL", base.confidence_threshold_sell),
        min_risk_reward=_env_float("AUTOGLUON_MIN_RR", base.min_risk_reward),
        min_volume_ratio=_env_float("AUTOGLUON_MIN_VOLUME_RATIO", base.min_volume_ratio),
        max_atr_percent_crypto=_env_float("AUTOGLUON_MAX_ATR_CRYPTO", base.max_atr_percent_crypto),
        max_atr_percent_idx=_env_float("AUTOGLUON_MAX_ATR_IDX", base.max_atr_percent_idx),
        horizon_candles=_env_int("AUTOGLUON_HORIZON_CANDLES", base.horizon_candles),
        embargo_candles=_env_int("AUTOGLUON_EMBARGO_CANDLES", base.embargo_candles),
        train_ratio=_env_float("AUTOGLUON_TRAIN_RATIO", base.train_ratio),
        tp_atr_multiplier=_env_float("AUTOGLUON_TP_ATR", base.tp_atr_multiplier),
        sl_atr_multiplier=_env_float("AUTOGLUON_SL_ATR", base.sl_atr_multiplier),
        min_train_rows=_env_int("AUTOGLUON_MIN_TRAIN_ROWS", base.min_train_rows),
        min_valid_rows=_env_int("AUTOGLUON_MIN_VALID_ROWS", base.min_valid_rows),
        min_symbol_candles_daily=_env_int("AUTOGLUON_MIN_DAILY_CANDLES", base.min_symbol_candles_daily),
        min_symbol_candles_intraday=_env_int("AUTOGLUON_MIN_INTRADAY_CANDLES", base.min_symbol_candles_intraday),
        max_symbols_per_batch=_env_int("AUTOGLUON_MAX_SYMBOLS", base.max_symbols_per_batch),
        model_dir=os.getenv("AUTOGLUON_MODEL_DIR", base.model_dir),
        training_dir=os.getenv("AUTOGLUON_TRAINING_DIR", base.training_dir),
        backtest_dir=os.getenv("AUTOGLUON_BACKTEST_DIR", base.backtest_dir),
        log_dir=os.getenv("AUTOGLUON_LOG_DIR", base.log_dir),
        use_gpu_fallback=_env_bool("AUTOGLUON_GPU_FALLBACK", base.use_gpu_fallback),
        min_precision_at_threshold=_env_float("AUTOGLUON_MIN_PRECISION_THRESHOLD", base.min_precision_at_threshold),
        min_backtest_signals=_env_int("AUTOGLUON_MIN_BACKTEST_SIGNALS", base.min_backtest_signals),
    )


AUTOGLUON_CONFIG = load_config().as_dict()


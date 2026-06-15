from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionGateConfig:
    min_trade_count: int = 5
    min_average_r: float = 0.05
    min_profit_factor: float = 1.05
    max_drawdown_r: float = -30.0
    min_precision_at_threshold: float = 0.50
    max_ece: float = 0.25


def evaluate_promotion_gate(metrics: dict, cfg: PromotionGateConfig) -> dict:
    gates = {
        "min_trade_count": metrics.get("trade_count_at_threshold", 0) >= cfg.min_trade_count,
        "min_average_r": metrics.get("average_r_at_threshold", -999) >= cfg.min_average_r,
        "min_profit_factor": metrics.get("profit_factor_at_threshold", 0) >= cfg.min_profit_factor,
        "max_drawdown_r": metrics.get("max_drawdown_r_at_threshold", -999) >= cfg.max_drawdown_r,
        "min_precision_at_threshold": metrics.get("precision_at_threshold", 0) >= cfg.min_precision_at_threshold,
        "max_ece": metrics.get("ece", 999) <= cfg.max_ece,
    }

    passed = all(gates.values())
    failed = [name for name, ok in gates.items() if not ok]

    return {
        "passed": passed,
        "failed_gates": failed,
        "gates": gates,
        "metrics": metrics,
    }

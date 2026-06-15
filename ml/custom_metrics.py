from __future__ import annotations

import numpy as np


def profit_factor_scorer_func(y_true, y_pred_proba, sample_weight=None, threshold: float = 0.75):
    """
    Experimental threshold-based profit-like scorer.

    WARNING:
    This is not the default training objective.
    It is optional because threshold-based profit metrics can be noisy.
    """
    y_true = np.asarray(y_true)

    proba = np.asarray(y_pred_proba)

    # AutoGluon / sklearn may pass probability array shape (n, 2).
    if proba.ndim == 2 and proba.shape[1] >= 2:
        proba = proba[:, 1]

    y_pred = (proba >= threshold).astype(int)
    mask = y_pred == 1

    if mask.sum() == 0:
        return 0.0

    wins = (y_true[mask] == 1).sum()
    losses = (y_true[mask] == 0).sum()

    if losses <= 0:
        return float(wins) if wins > 0 else 0.0

    return float(wins / losses)


def build_optional_profit_scorer():
    try:
        from autogluon.core.metrics import make_scorer
    except Exception:
        return None

    return make_scorer(
        name="custom_profit_metric",
        score_func=profit_factor_scorer_func,
        optimum=float("inf"),
        greater_is_better=True,
        needs_pred=True,
        needs_threshold=True,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class PurgedSplitConfig:
    train_ratio: float = 0.8
    horizon_candles: int = 20
    embargo_candles: int = 20
    min_train_rows: int = 200
    min_valid_rows: int = 50


def purged_embargo_split(
    df: pd.DataFrame,
    cfg: PurgedSplitConfig,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation split for horizon-based labels.

    If label at row i uses future candles i+1...i+horizon,
    then training rows near the split must be purged and validation
    must start after an embargo gap.

    This prevents training labels from depending on validation-period prices.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    if not 0.1 <= cfg.train_ratio <= 0.95:
        raise ValueError(f"Invalid train_ratio: {cfg.train_ratio}")

    if cfg.horizon_candles < 1:
        raise ValueError("horizon_candles must be >= 1")

    if cfg.embargo_candles < 0:
        raise ValueError("embargo_candles must be >= 0")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(ordered)

    if n < cfg.min_train_rows + cfg.min_valid_rows + cfg.horizon_candles + cfg.embargo_candles:
        raise ValueError(
            "Not enough rows for purged/embargo split: "
            f"rows={n}, min_train={cfg.min_train_rows}, "
            f"min_valid={cfg.min_valid_rows}, horizon={cfg.horizon_candles}, "
            f"embargo={cfg.embargo_candles}"
        )

    split_idx = int(n * cfg.train_ratio)

    train_end = split_idx - cfg.horizon_candles
    valid_start = split_idx + cfg.embargo_candles

    if train_end <= 0:
        raise ValueError("Purged train set is empty. Reduce horizon or train_ratio.")

    if valid_start >= n:
        raise ValueError("Validation set is empty. Reduce embargo or train_ratio.")

    train_df = ordered.iloc[:train_end].copy()
    valid_df = ordered.iloc[valid_start:].copy()

    if len(train_df) < cfg.min_train_rows:
        raise ValueError(f"Train rows below minimum: {len(train_df)} < {cfg.min_train_rows}")

    if len(valid_df) < cfg.min_valid_rows:
        raise ValueError(f"Validation rows below minimum: {len(valid_df)} < {cfg.min_valid_rows}")

    return train_df, valid_df

from __future__ import annotations

from typing import Iterable, List


FORBIDDEN_FEATURE_PATTERNS = [
    "future",
    "lookahead",
    "lead_",
    "shift_minus",
    "next_",
    "tp_hit",
    "sl_hit",
    "tp_before_sl",
    "hit_tp",
    "hit_sl",
    "outcome",
    "r_multiple",
    "max_favorable",
    "max_adverse",
    "mfe",
    "mae",
    "expired_after",
    "invalidated_after",
    "entry_touched_after",
    "direction_5",
    "direction_10",
]


def assert_no_leaky_features(feature_columns: Iterable[str], label: str | None = None) -> List[str]:
    """
    Raise if feature columns contain future/outcome-derived fields.
    Returns clean feature list if valid.
    """
    clean = []
    violations = []

    label_lower = label.lower() if label else None

    for col in feature_columns:
        c = str(col)
        lower = c.lower()

        if label_lower and lower == label_lower:
            violations.append(c)
            continue

        for pattern in FORBIDDEN_FEATURE_PATTERNS:
            if pattern in lower:
                violations.append(c)
                break
        else:
            clean.append(c)

    if violations:
        raise ValueError(
            "Future/outcome leakage columns found in features: "
            + ", ".join(sorted(set(violations)))
        )

    return clean

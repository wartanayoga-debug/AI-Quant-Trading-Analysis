from __future__ import annotations

import json
from typing import Any, Dict, List

from .inference_engine import screen_ohlcv
from .model_registry import get_model_metadata, list_all_models, rollback_model
from .train_autogluon import gpu_sanity_check, train_from_csv
from .walk_forward_backtest import walk_forward_backtest


def status() -> Dict[str, Any]:
    return {
        "ok": True,
        "gpu": gpu_sanity_check(),
        "latest_model": get_model_metadata("latest"),
        "all_models": list_all_models(),
    }


def train(csv_path: str, target: str = "target_buy_valid") -> Dict[str, Any]:
    return train_from_csv(csv_path, target)


def screen(candles: List[Dict[str, Any]], side: str = "BUY") -> Dict[str, Any]:
    return screen_ohlcv(candles, side)


def backtest(csv_path: str, target: str = "target_buy_valid", folds: int = 3) -> Dict[str, Any]:
    return walk_forward_backtest(csv_path, target, folds)


def rollback(version: str) -> Dict[str, Any]:
    return rollback_model(version)


def delete(version: str) -> Dict[str, Any]:
    from .model_registry import delete_model
    return delete_model(version)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AutoGluon engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    p_train = sub.add_parser("train")
    p_train.add_argument("--csv", required=True)
    p_train.add_argument("--target", default="target_buy_valid")
    p_backtest = sub.add_parser("backtest")
    p_backtest.add_argument("--csv", required=True)
    p_backtest.add_argument("--target", default="target_buy_valid")
    p_backtest.add_argument("--folds", type=int, default=3)
    p_rollback = sub.add_parser("rollback")
    p_rollback.add_argument("--version", required=True)
    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--version", required=True)
    p_screen = sub.add_parser("screen")
    p_screen.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    args = parser.parse_args()

    if args.command == "status":
        result = status()
    elif args.command == "train":
        result = train(args.csv, args.target)
    elif args.command == "backtest":
        result = backtest(args.csv, args.target, args.folds)
    elif args.command == "rollback":
        result = rollback(args.version)
    elif args.command == "delete":
        result = delete(args.version)
    elif args.command == "screen":
        import sys

        payload = json.loads(sys.stdin.read() or "{}")
        result = screen(payload.get("candles", []), args.side)
    else:
        result = {"ok": False, "error": "unknown_command"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

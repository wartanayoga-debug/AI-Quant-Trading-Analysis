from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config


def _model_root() -> Path:
    root = Path(load_config().model_dir)
    (root / "versions").mkdir(parents=True, exist_ok=True)
    (root / "latest").mkdir(parents=True, exist_ok=True)
    return root


def get_latest_model_path() -> Optional[Path]:
    latest_meta = _model_root() / "latest" / "metadata.json"
    if not latest_meta.exists():
        return None
    meta = json.loads(latest_meta.read_text(encoding="utf-8"))
    path = _resolve_model_path(meta)
    return path if path.exists() else None


def get_model_metadata(version: str = "latest") -> Dict[str, Any]:
    root = _model_root()
    meta_path = root / version / "metadata.json" if version in ["latest", "challenger"] else root / "versions" / version / "metadata.json"
    if not meta_path.exists():
        return {"available": False, "reason": "model_metadata_not_found"}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = _resolve_model_path(data)
    data["model_path"] = str(model_path) if model_path else data.get("model_path", "")
    data["available"] = bool(model_path and model_path.exists())
    if not data["available"]:
        data["reason"] = "model_path_not_found"
    return data


def register_model(version: str, metrics: Dict[str, Any], model_path: str | Path, alias: str = None) -> Dict[str, Any]:
    root = _model_root()
    model_path = Path(model_path)
    version_dir = root / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": version,
        "model_path": str(model_path.resolve()),
        "metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if alias:
        set_alias(version, alias)
    return metadata


def set_alias(version: str, alias: str) -> None:
    root = _model_root()
    version_dir = root / "versions" / version
    alias_dir = root / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    meta_path = version_dir / "metadata.json"
    if meta_path.exists():
        import shutil
        shutil.copy2(meta_path, alias_dir / "metadata.json")


def compare_model_with_current(candidate_metrics: Dict[str, Any]) -> bool:
    current = get_model_metadata("latest")
    if not current.get("available"):
        return True
    old_metrics = current.get("metrics", {})
    old_precision = float(old_metrics.get("precision_at_0_75", 0) or 0)
    new_precision = float(candidate_metrics.get("precision_at_0_75", 0) or 0)
    old_auc = float(old_metrics.get("roc_auc", 0) or 0)
    new_auc = float(candidate_metrics.get("roc_auc", 0) or 0)
    return (new_precision, new_auc) >= (old_precision, old_auc)


def rollback_model(version: str) -> Dict[str, Any]:
    root = _model_root()
    meta_path = root / "versions" / version / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model version not found: {version}")
    latest_dir = root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, latest_dir / "metadata.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def delete_model(version: str) -> Dict[str, Any]:
    current = get_model_metadata("latest")
    if current.get("version") == version:
        raise ValueError(f"Cannot delete active model version: {version}")
    root = _model_root()
    version_dir = root / "versions" / version
    if not version_dir.exists():
        raise FileNotFoundError(f"model version not found: {version}")
    shutil.rmtree(version_dir)
    return {"ok": True, "deleted_version": version}


def load_model(version: str = "latest"):
    if version == "latest":
        path = get_latest_model_path()
    else:
        metadata = get_model_metadata(version)
        path = _resolve_model_path(metadata)
    if path is None or not Path(path).exists():
        raise FileNotFoundError("AutoGluon latest model is not available.")
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise RuntimeError("AutoGluon is not installed. Install autogluon to load models.") from exc
    return TabularPredictor.load(str(path))


def list_all_models() -> List[Dict[str, Any]]:
    root = _model_root()
    versions_dir = root / "versions"
    if not versions_dir.exists():
        return []
    
    models = []
    for d in versions_dir.iterdir():
        if d.is_dir():
            meta_path = d / "metadata.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    model_path = _resolve_model_path(data)
                    data["model_path"] = str(model_path) if model_path else data.get("model_path", "")
                    data["available"] = bool(model_path and model_path.exists())
                    models.append(data)
                except Exception:
                    pass
    
    # Sort by registered_at descending
    models.sort(key=lambda x: x.get("registered_at", ""), reverse=True)
    return models


def _resolve_model_path(metadata: Dict[str, Any]) -> Optional[Path]:
    raw_path = metadata.get("model_path")
    if raw_path:
        direct = Path(raw_path)
        predictor = direct / "predictor"
        if predictor.exists():
            return predictor
        if direct.exists():
            return direct

    version = metadata.get("version")
    if version:
        version_dir = _model_root() / "versions" / str(version)
        predictor = version_dir / "predictor"
        if predictor.exists():
            return predictor
        if version_dir.exists():
            return version_dir

    return None

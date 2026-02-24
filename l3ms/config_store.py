from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "model_downloader" / "models_config.json"
VERSIONS_ROOT = ROOT / ".toolkit" / "download_config_versions"


MODEL_KEYS = {
    "enabled",
    "repo_id",
    "local_dir",
    "allow_patterns",
    "ignore_patterns",
    "revision",
    "force_download",
    "preserve_existing",
    "max_workers",
    "description",
}


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _safe_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _version_dir_for(config_path: Path) -> Path:
    key = re.sub(r"[^a-zA-Z0-9._/-]+", "-", str(config_path.resolve())).strip("/")
    key = key.replace("/", "__") or "models_config_json"
    return VERSIONS_ROOT / key


def normalize_model(model: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "enabled": bool(model.get("enabled", True)),
        "repo_id": str(model.get("repo_id", "")).strip(),
        "local_dir": str(model.get("local_dir", "")).strip(),
        "allow_patterns": [str(p).strip() for p in (model.get("allow_patterns") or []) if str(p).strip()],
        "ignore_patterns": [str(p).strip() for p in (model.get("ignore_patterns") or []) if str(p).strip()],
        "revision": str(model.get("revision", "")).strip(),
        "force_download": bool(model.get("force_download", False)),
        "preserve_existing": bool(model.get("preserve_existing", True)),
        "max_workers": model.get("max_workers"),
        "description": str(model.get("description", "")).strip(),
    }
    if not isinstance(out["max_workers"], int) or out["max_workers"] <= 0:
        out["max_workers"] = None
    return out


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw_models = raw.get("models") if isinstance(raw, dict) else []
    models: List[Dict[str, Any]] = []
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, dict):
                filtered = {k: v for k, v in item.items() if k in MODEL_KEYS}
                models.append(normalize_model(filtered))

    return {
        "base_models_dir": str(raw.get("base_models_dir", "")).strip() if isinstance(raw, dict) else "",
        "models": models,
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(config, dict):
        return ["Config must be a JSON object"]

    models = config.get("models")
    if not isinstance(models, list):
        errors.append("models must be an array")
        return errors

    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{i}] must be an object")
            continue

        if not str(model.get("repo_id", "")).strip():
            errors.append(f"models[{i}].repo_id is required")

        for key in ("allow_patterns", "ignore_patterns"):
            val = model.get(key, [])
            if val is None:
                continue
            if not isinstance(val, list):
                errors.append(f"models[{i}].{key} must be an array")
                continue
            if any(not isinstance(x, str) for x in val):
                errors.append(f"models[{i}].{key} entries must be strings")

        workers = model.get("max_workers")
        if workers is not None and (not isinstance(workers, int) or workers <= 0):
            errors.append(f"models[{i}].max_workers must be null or positive integer")

    return errors


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"base_models_dir": "", "models": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"base_models_dir": "", "models": []}
    if not isinstance(raw, dict):
        return {"base_models_dir": "", "models": []}
    return normalize_config(raw)


def save_config(path: Path, config: Dict[str, Any], note: str = "manual") -> None:
    normalized = normalize_config(config)
    errors = validate_config(normalized)
    if errors:
        raise ValueError("; ".join(errors))

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        VERSIONS_ROOT.mkdir(parents=True, exist_ok=True)
        version_dir = _version_dir_for(path)
        version_dir.mkdir(parents=True, exist_ok=True)
        safe_note = re.sub(r"[^a-zA-Z0-9._-]+", "-", note).strip("-") or "save"
        backup_name = f"{_safe_stamp()}__{safe_note}.json"
        backup = version_dir / backup_name
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def list_versions(path: Path) -> List[str]:
    version_dir = _version_dir_for(path.resolve())
    if not version_dir.exists():
        return []
    return sorted([p.name for p in version_dir.iterdir() if p.is_file() and p.suffix == ".json"], reverse=True)


def restore_version(path: Path, version_name: str) -> None:
    source = _version_dir_for(path.resolve()) / version_name
    if not source.exists() or not source.is_file():
        raise ValueError("version not found")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def csv_to_list(raw: str) -> List[str]:
    return _split_csv(raw)

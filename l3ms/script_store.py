from __future__ import annotations

import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = ROOT / ".toolkit" / "script_versions"
ALLOWED_EXTENSIONS = {".sh", ".ps1", ".bat", ".cmd"}


def command_for_script(path: Path, extra_args: List[str]) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".sh":
        return ["bash", str(path), *extra_args]
    if suffix == ".ps1":
        return ["pwsh", "-File", str(path), *extra_args]
    if suffix in {".bat", ".cmd"}:
        return ["cmd", "/c", str(path), *extra_args]
    return ["bash", str(path), *extra_args]


def _safe_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_note(note: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", note).strip("-")
    return clean[:40] if clean else "save"


def _relative_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("script path must stay inside repository") from exc


def resolve_script(path: Path) -> Path:
    target = path.resolve()
    _relative_key(target)
    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("unsupported script extension")
    return target


def version_dir_for_script(path: Path) -> Path:
    rel = _relative_key(resolve_script(path))
    return VERSIONS_ROOT / rel


def list_script_versions(path: Path) -> List[str]:
    version_dir = version_dir_for_script(path)
    if not version_dir.exists():
        return []
    return sorted([p.name for p in version_dir.iterdir() if p.is_file()], reverse=True)


def load_script(path: Path) -> str:
    target = resolve_script(path)
    if not target.exists() or not target.is_file():
        raise ValueError("script not found")
    return target.read_text(encoding="utf-8")


def save_script_with_version(path: Path, content: str, note: str = "manual") -> None:
    target = resolve_script(path)
    if target.exists() and target.is_file():
        previous = target.read_text(encoding="utf-8")
        mode = stat.S_IMODE(target.stat().st_mode)
    else:
        previous = ""
        mode = 0o755

    version_dir = version_dir_for_script(target)
    version_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"{_safe_stamp()}__{_sanitize_note(note)}{target.suffix}"
    (version_dir / backup_name).write_text(previous, encoding="utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    target.chmod(mode)


def restore_script_version(path: Path, version_name: str) -> None:
    target = resolve_script(path)
    version_path = (version_dir_for_script(target) / version_name).resolve()
    if not version_path.exists() or not version_path.is_file():
        raise ValueError("version not found")
    if version_dir_for_script(target).resolve() not in version_path.parents:
        raise ValueError("invalid version path")
    content = version_path.read_text(encoding="utf-8")
    save_script_with_version(target, content, note=f"restore-{version_name}")

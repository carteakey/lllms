#!/home/kchauhan/repos/l3ms/.venv/bin/python3
"""
HuggingFace Model Downloader

Downloads models from HuggingFace Hub with configurable patterns and concurrency.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repo root (two levels up from this file)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print('Please install it with: pip install -U "huggingface_hub"')
    sys.exit(1)


# Detect active download backend and report it.
# hf_xet is the current fast path (chunk-based deduplication via Xet storage).
# hf_transfer is deprecated and must NOT be force-enabled alongside hf_xet.
def _report_download_backend() -> None:
    try:
        import importlib.metadata as _meta

        xet_ver = _meta.version("hf_xet")
        print(f"  download backend: hf_xet {xet_ver} (Xet storage, chunk dedup)")
        return
    except Exception:
        pass
    # Warn if someone still has HF_HUB_ENABLE_HF_TRANSFER set in environment
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print(
            "  download backend: hf_transfer (DEPRECATED — unset HF_HUB_ENABLE_HF_TRANSFER and install hf_xet)"
        )
        return
    print(
        "  download backend: huggingface_hub default (install hf_xet for faster downloads)"
    )


def has_local_files(local_dir: str) -> bool:
    """Return True if the local directory contains at least one non-hidden file."""
    local = Path(local_dir)
    return local.exists() and any(
        f for f in local.rglob("*") if f.is_file() and not f.name.startswith(".")
    )


def download_model(
    repo_id: str,
    local_dir: str,
    allow_patterns: List[str] = None,
    ignore_patterns: List[str] = None,
    revision: str = None,
    force_download: bool = False,
    max_workers: Optional[int] = None,
    update_only: bool = False,
) -> str:
    if update_only:
        if not has_local_files(local_dir):
            print(
                f"  No local files found — skipping (run without --update to download fresh)"
            )
            return local_dir
        print(f"  ↻ Syncing {repo_id} — pulling only changed/new files")

    print(f"Downloading {repo_id} to {local_dir}")
    if allow_patterns:
        print(f"  Including patterns: {allow_patterns}")
    if ignore_patterns:
        print(f"  Excluding patterns: {ignore_patterns}")
    if max_workers is not None:
        print(f"  Max download workers: {max_workers}")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision or "main",
            force_download=force_download,
            max_workers=max_workers,
        )
        print(f"✓ Successfully downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as hub_err:
        local = Path(local_dir)
        if local.exists() and any(f for f in local.iterdir() if f.is_file()):
            print(f"  Hub unreachable, keeping existing files in {local_dir}")
            print(f"  (Reason: {hub_err})")
            return local_dir
        print(f"✗ Error downloading {repo_id}: {hub_err}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}


def resolve_local_dir(repo_id: str, base_models_dir: str, local_dir: str = None) -> str:
    if local_dir:
        return local_dir
    parts = repo_id.split("/")
    if len(parts) == 2:
        org, model = parts
        return os.path.join(base_models_dir, org, model)
    return os.path.join(base_models_dir, repo_id.replace("/", "_"))


def main():
    _report_download_backend()
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"
  ./download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --max-workers 2
  ./download_hf_model.py --config models_config.json
  ./download_hf_model.py --config models_config.json --slow
  ./download_hf_model.py --config models_config.json --update
        """,
    )

    parser.add_argument("--config", "-c", help="Path to JSON configuration file")
    parser.add_argument("--repo-id", "-r", help="Repository ID to download")
    parser.add_argument("--local-dir", "-d", help="Local directory to save the model")
    parser.add_argument(
        "--allow-patterns", "-a", nargs="+", help="File patterns to include"
    )
    parser.add_argument(
        "--ignore-patterns", "-i", nargs="+", help="File patterns to exclude"
    )
    parser.add_argument("--revision", help="Specific revision/branch to download")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Max concurrent download workers"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Slow preset: max_workers=4"
    )
    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Sync updates for models already on disk; skip models with no local files",
    )
    parser.add_argument("--base-models-dir", help="Base directory for all models")

    args = parser.parse_args()
    effective_max_workers = (
        args.max_workers if args.max_workers is not None else (4 if args.slow else None)
    )
    base_models_dir = args.base_models_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )

    if args.config:
        config = load_config(args.config)
        if not config:
            return

        base_models_dir = config.get("base_models_dir", base_models_dir)
        models = config.get("models", [])
        enabled_models = [m for m in models if m.get("enabled", True)]
        disabled_count = len(models) - len(enabled_models)

        print(
            f"Found {len(models)} models in configuration ({len(enabled_models)} enabled, {disabled_count} disabled)"
        )

        if not enabled_models:
            print("No enabled models to download")
            return

        for i, m in enumerate(enabled_models, 1):
            repo_id = m.get("repo_id")
            if not repo_id:
                print(f"Skipping model {i}: no repo_id specified")
                continue

            print(f"\n[{i}/{len(enabled_models)}] Processing {repo_id}")
            if m.get("description"):
                print(f"  Description: {m['description']}")

            local_dir = resolve_local_dir(repo_id, base_models_dir, m.get("local_dir"))

            try:
                download_model(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    allow_patterns=m.get("allow_patterns"),
                    ignore_patterns=m.get("ignore_patterns"),
                    revision=m.get("revision"),
                    force_download=m.get("force_download", False),
                    max_workers=m.get("max_workers", effective_max_workers),
                    update_only=args.update,
                )
            except Exception as e:
                print(f"Failed to download {repo_id}: {e}")
                continue

    elif args.repo_id:
        local_dir = resolve_local_dir(args.repo_id, base_models_dir, args.local_dir)
        download_model(
            repo_id=args.repo_id,
            local_dir=local_dir,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            revision=args.revision,
            force_download=args.force_download,
            max_workers=effective_max_workers,
            update_only=args.update,
        )

    else:
        parser.print_help()
        print("\nError: Either --repo-id or --config must be specified")


if __name__ == "__main__":
    main()

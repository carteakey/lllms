#!/usr/bin/env python3
"""
Dynamic HuggingFace Model Downloader

This script provides a flexible way to download models from HuggingFace Hub
with configurable options for repo IDs, local directories, and file patterns.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub hf_transfer")
    sys.exit(1)


class ModelDownloader:
    """Handles downloading models from HuggingFace Hub."""

    def __init__(self, base_models_dir: str = None):
        self.base_models_dir = base_models_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )
        Path(self.base_models_dir).mkdir(parents=True, exist_ok=True)

    def download_model(
        self,
        repo_id: str,
        local_dir: str = None,
        allow_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
        revision: str = None,
        force_download: bool = False
    ) -> str:
        """
        Download a model from HuggingFace Hub.

        Args:
            repo_id: The repository ID (e.g., "microsoft/DialoGPT-medium")
            local_dir: Local directory to save the model (auto-generated if None)
            allow_patterns: List of file patterns to include (e.g., ["*.bin", "*.json"])
            ignore_patterns: List of file patterns to exclude
            revision: Specific revision/branch to download
            force_download: Whether to re-download existing files

        Returns:
            str: Path to the downloaded model directory
        """
        if local_dir is None:
            # Auto-generate local directory based on repo_id
            repo_parts = repo_id.split("/")
            if len(repo_parts) == 2:
                org, model = repo_parts
                local_dir = os.path.join(self.base_models_dir, org, model)
            else:
                local_dir = os.path.join(self.base_models_dir, repo_id.replace("/", "_"))

        print(f"Downloading {repo_id} to {local_dir}")
        if allow_patterns:
            print(f"  Including patterns: {allow_patterns}")
        if ignore_patterns:
            print(f"  Excluding patterns: {ignore_patterns}")

        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                revision=revision,
                force_download=force_download
            )
            print(f"✓ Successfully downloaded to: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"✗ Error downloading {repo_id}: {e}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}


def create_sample_config(config_path: str):
    """Create a sample configuration file."""
    sample_config = {
        "base_models_dir": "./models",
        "models": [
            {
                "repo_id": "microsoft/DialoGPT-medium",
                "allow_patterns": ["*.bin", "*.json", "*.txt"],
                "description": "DialoGPT medium model"
            },
            {
                "repo_id": "Qwen/Qwen3-32B-GGUF",
                "allow_patterns": ["*Q6_K*"],
                "local_dir": "./models/qwen/Qwen3-32B-GGUF",
                "description": "Qwen3 32B model with Q6_K quantization"
            },
            {
                "repo_id": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                "allow_patterns": ["*Q8*"],
                "description": "Qwen3 30B Instruct with Q8 quantization"
            }
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    print(f"Sample configuration created at: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a single model
  python download_hf_model.py --repo-id microsoft/DialoGPT-medium

  # Download with specific patterns
  python download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"

  # Download to specific directory
  python download_hf_model.py --repo-id microsoft/DialoGPT-medium --local-dir ./my_models/dialogs

  # Use configuration file
  python download_hf_model.py --config models_config.json

  # Create sample config
  python download_hf_model.py --create-config models_config.json
        """
    )

    # Configuration options
    parser.add_argument(
        "--config", "-c",
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--create-config",
        help="Create a sample configuration file at the specified path"
    )

    # Single model download options
    parser.add_argument(
        "--repo-id", "-r",
        help="Repository ID to download (e.g., microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--local-dir", "-d",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--allow-patterns", "-a",
        nargs="+",
        help="File patterns to include (e.g., *.bin *.json)"
    )
    parser.add_argument(
        "--ignore-patterns", "-i",
        nargs="+",
        help="File patterns to exclude"
    )
    parser.add_argument(
        "--revision",
        help="Specific revision/branch to download"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download existing files"
    )
    parser.add_argument(
        "--base-models-dir",
        help="Base directory for all models (default: ./models)"
    )

    args = parser.parse_args()

    # Create sample config and exit
    if args.create_config:
        create_sample_config(args.create_config)
        return

    # Initialize downloader
    downloader = ModelDownloader(args.base_models_dir)

    # Download from config file
    if args.config:
        config = load_config(args.config)
        if not config:
            return

        # Update base directory if specified in config
        if "base_models_dir" in config:
            downloader.base_models_dir = config["base_models_dir"]

        models = config.get("models", [])
        if not models:
            print("No models found in configuration file")
            return

        print(f"Found {len(models)} models in configuration")
        for i, model_config in enumerate(models, 1):
            repo_id = model_config.get("repo_id")
            if not repo_id:
                print(f"Skipping model {i}: no repo_id specified")
                continue

            print(f"\n[{i}/{len(models)}] Processing {repo_id}")
            if "description" in model_config:
                print(f"  Description: {model_config['description']}")

            try:
                downloader.download_model(
                    repo_id=repo_id,
                    local_dir=model_config.get("local_dir"),
                    allow_patterns=model_config.get("allow_patterns"),
                    ignore_patterns=model_config.get("ignore_patterns"),
                    revision=model_config.get("revision"),
                    force_download=model_config.get("force_download", False)
                )
            except Exception as e:
                print(f"Failed to download {repo_id}: {e}")
                continue

    # Download single model
    elif args.repo_id:
        downloader.download_model(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            revision=args.revision,
            force_download=args.force_download
        )

    else:
        parser.print_help()
        print("\nError: Either --repo-id or --config must be specified")


if __name__ == "__main__":
    main()

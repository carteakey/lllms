#!/usr/bin/env python3
import json
import yaml
import os
import glob
from pathlib import Path

def audit():
    repo_root = Path(__file__).parent.parent.parent.parent
    config_path = repo_root / "model_downloader/models_config.json"
    swap_path = repo_root / "llama-swap.yaml"
    models_dir = Path("/mnt/lab/models")

    # 1. Load configs
    with open(config_path, 'r') as f:
        downloader_config = json.load(f)
    
    with open(swap_path, 'r') as f:
        swap_config = yaml.safe_load(f)

    # 2. Map served models
    served_paths = set()
    for m_id, m_cfg in swap_config.get('models', {}).items():
        cmd = m_cfg.get('cmd', '')
        # Simple extraction of paths from cmd string
        parts = cmd.split()
        for i, p in enumerate(parts):
            if p in ('-m', '--model', '--mmproj', '--mtp-head'):
                if i + 1 < len(parts):
                    served_paths.add(os.path.abspath(parts[i+1].replace("${env.L3MS_ROOT}", str(repo_root))))

    # 3. Scan disk
    disk_files = set(glob.glob(str(models_dir / "**/*.gguf"), recursive=True))
    disk_files = {os.path.abspath(f) for f in disk_files}

    # 4. Downloader inventory
    registered_dirs = {os.path.abspath(m['local_dir']) for m in downloader_config['models']}
    disabled_dirs = {os.path.abspath(m['local_dir']) for m in downloader_config['models'] if not m.get('enabled', True)}

    print("--- L3MS Model Inventory Audit ---")

    # Check Broken Serving Configs
    broken_served = [p for p in served_paths if not os.path.exists(p)]
    if broken_served:
        print("\n[!] BROKEN SERVING CONFIGS (Files missing but referenced in llama-swap.yaml):")
        for p in broken_served:
            print(f"  - {p}")

    # Check Orphans (On disk but not in serving config)
    orphans = []
    for f in disk_files:
        if f not in served_paths:
            # Check if it's a known non-serving model (image/video)
            is_known_non_serving = False
            for d in registered_dirs:
                if f.startswith(d) and ("Image" in d or "LTX" in d):
                    is_known_non_serving = True
                    break
            if not is_known_non_serving:
                orphans.append(f)

    if orphans:
        print("\n[?] POTENTIAL ORPHANS (On disk but not in llama-swap.yaml):")
        for f in sorted(orphans):
            print(f"  - {f}")

    # Check Disabled but present
    dangling_disabled = []
    for d in disabled_dirs:
        if os.path.exists(d) and os.listdir(d):
            dangling_disabled.append(d)
    
    if dangling_disabled:
        print("\n[!] DANGLING DISABLED MODELS (Disabled in downloader but still on disk):")
        for d in dangling_disabled:
            print(f"  - {d}")

    # Unregistered models
    unregistered = []
    for f in disk_files:
        is_registered = False
        for d in registered_dirs:
            if f.startswith(d):
                is_registered = True
                break
        if not is_registered:
            unregistered.append(f)
    
    if unregistered:
        print("\n[!] UNREGISTERED MODELS (On disk but not in models_config.json):")
        for f in sorted(unregistered):
            print(f"  - {f}")

if __name__ == "__main__":
    audit()

---
name: model-inventory-manager
description: Audit and cleanup model files and configurations. Use when identifying old models, cleaning up disk space, or reconciling serving configs with physical files.
---

# Model Inventory Manager

This skill provides workflows for maintaining a lean and functional model storage system.

## Workflow

### 1. Audit Inventory
Run the audit script to identify discrepancies between `llama-swap.yaml`, `models_config.json`, and the filesystem.

```bash
python3 codex-skills/model-inventory-manager/scripts/audit_models.py
```

The script categorizes findings into:
- **Broken Serving Configs**: Referenced in YAML but missing from disk.
- **Potential Orphans**: On disk but not referenced in the serving config (excluding known non-serving types like Image/Video).
- **Dangling Disabled Models**: Disabled in the downloader but still occupying space.
- **Unregistered Models**: On disk but not tracked in the downloader config.

### 2. Reconciliation
- **Fix Broken Configs**: Remove or update entries in `llama-swap.yaml` that point to missing files.
- **Cleanup Orphans**: Verify if orphaned files are truly unneeded before deletion.
- **Disable Downloads**: Ensure unused models are set to `"enabled": false` in `model_downloader/models_config.json`.

### 3. Preservation Rules
Always preserve the following unless explicitly instructed:
- `LTX` (Video generation)
- `Image-Edit` / `Image-2512` (Image processing)
- Production defaults like `gpt-oss-120b` and `gemma-4-26b-mtp`.

# /download-model

Trigger a targeted model download from HuggingFace using the L3MS downloader.
Prints the command to run; does not execute automatically to preserve explicit control.

## Usage

```
/download-model <hf-repo-id> <quant-pattern> [--local-dir <path>] [--workers <N>]
```

**Arguments:**
- `<hf-repo-id>` — HuggingFace repo ID (e.g. `unsloth/gemma-4-26B-A4B-it-GGUF`)
- `<quant-pattern>` — glob pattern for files to download (e.g. `*UD-Q5_K_XL*`)
- `--local-dir <path>` — override local storage path (default: `/mnt/lab/models/<hf-repo-id>`)
- `--workers <N>` — parallel download workers (default: 2)

## Workflow

### Step 1 — Check if model already exists

```bash
ls /mnt/lab/models/<hf-repo-id>/
# or check the default path from models_config.json
```

Models are preserved by default (`force_download: false`). If the file exists, the download will skip it.

### Step 2 — Check models_config.json for an existing profile

Read `model_downloader/models_config.json`:
```bash
cat model_downloader/models_config.json | jq '.[] | select(.repo_id == "<hf-repo-id>") | {name, local_dir, allow_patterns}'
```

If a profile exists, use its `local_dir` and `allow_patterns` unless overridden.

### Step 3 — Generate and print the download command

**Targeted download (preferred):**
```bash
./model_downloader/download_hf_model.py \
  --repo-id <hf-repo-id> \
  --allow-patterns '<quant-pattern>' \
  --local-dir /mnt/lab/models/<hf-repo-id> \
  --max-workers 2
```

**From config profile (if enabled in models_config.json):**
```bash
python3 model_downloader/download_hf_model.py --config-name <profile-name>
```

### Step 4 — Enable profile in config (optional)

If you want to add this download to the persistent config profile, update `model_downloader/models_config.json`:
- Set `"enabled": true` for the model entry
- Verify other entries are still `"enabled": false` to avoid accidental bulk downloads

### Download principles (from AGENTS.md)

- **Preserve existing files by default** — `force_download: false` is the safe default
- **Never enable all config rows** — always use targeted downloads or single enabled profiles
- **Use max_workers=2** for multi-file sharded models to avoid HF rate limits

## Common quant patterns by repo type

| Model family | Recommended quant | Pattern |
|-------------|-------------------|---------|
| Unsloth UD quants | UD-Q5_K_XL or UD-Q4_K_XL | `*UD-Q5_K_XL*` or `*UD-Q4_K_XL*` |
| Unsloth UD IQ | UD-IQ4_XS | `*UD-IQ4_XS*` |
| ggml-org MXFP4 | MXFP4_MOE | `*MXFP4_MOE*` |
| Unsloth mmproj | BF16 projector | `*mmproj-BF16*` |
| Mistral/vision mmproj | F16 projector | `*mmproj-F16*` |

## Disk space check

Before downloading, check available space on the target mount:
```bash
df -h /mnt/lab/models
# or for /home-based paths:
df -h ~/models
```

Typical model sizes:
| Model | Quant | Size |
|-------|-------|------|
| Qwen3-Coder-Next (80B) | UD-Q4_K_XL | ~47 GB |
| Qwen3.6-35B-A3B | UD-Q5_K_XL | ~25 GB |
| gpt-oss-120b | MXFP4 | ~59 GB |
| Gemma 4 26B-A4B | UD-Q5_K_XL | ~18 GB |
| Qwen3.5-122B | UD-IQ4_XS | ~56 GB |

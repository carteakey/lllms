# Dynamic Model Downloader

A flexible and configurable tool for downloading models from HuggingFace Hub with support for patterns, custom directories, and batch downloads.

## Features

- **Dynamic Configuration**: Use JSON config files or command-line arguments
- **Pattern Matching**: Include/exclude specific file patterns (e.g., `*Q8*`, `*.bin`)
- **Batch Downloads**: Download multiple models in one go
- **Auto-directory Generation**: Automatically organize models by repository structure
- **Fast Downloads**: Uses `hf_transfer` for improved download speeds
- **Error Handling**: Robust error handling with detailed feedback

## Installation

First, install the required dependencies:

```bash
pip install huggingface_hub hf_transfer
```

## Usage

### 1. Single Model Download

Download a single model with command-line arguments:

```bash
# Basic download
python download_hf_model.py --repo-id microsoft/DialoGPT-medium

# Download with specific patterns
python download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"

# Download to custom directory
python download_hf_model.py --repo-id microsoft/DialoGPT-medium --local-dir ./my_models/dialog

# Download specific revision/branch
python download_hf_model.py --repo-id microsoft/DialoGPT-medium --revision main

# Force re-download existing files
python download_hf_model.py --repo-id microsoft/DialoGPT-medium --force-download
```

### 2. Batch Download with Configuration

Create a configuration file and download multiple models:

```bash
# Use existing config
python download_hf_model.py --config models_config.json

# Create a sample config file
python download_hf_model.py --create-config my_models.json
```

### 3. Configuration File Format

```json
{
  "base_models_dir": "./models",
  "models": [
    {
      "repo_id": "microsoft/DialoGPT-medium",
      "allow_patterns": ["*.bin", "*.json", "*.txt"],
      "description": "DialoGPT medium conversational model"
    },
    {
      "repo_id": "Qwen/Qwen3-32B-GGUF",
      "local_dir": "./models/qwen/Qwen3-32B-GGUF",
      "allow_patterns": ["*Q6_K*"],
      "ignore_patterns": ["*.md"],
      "description": "Qwen3 32B with Q6_K quantization"
    }
  ]
}
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--repo-id` | `-r` | Repository ID to download |
| `--local-dir` | `-d` | Local directory to save the model |
| `--allow-patterns` | `-a` | File patterns to include |
| `--ignore-patterns` | `-i` | File patterns to exclude |
| `--config` | `-c` | Path to JSON configuration file |
| `--create-config` | | Create sample configuration file |
| `--revision` | | Specific revision/branch to download |
| `--force-download` | | Re-download existing files |
| `--base-models-dir` | | Base directory for all models |

## Configuration Options

### Model Configuration

Each model in the configuration can have these properties:

- `repo_id` (required): HuggingFace repository ID
- `local_dir` (optional): Custom local directory path
- `allow_patterns` (optional): List of file patterns to include
- `ignore_patterns` (optional): List of file patterns to exclude
- `revision` (optional): Specific git revision/branch/tag
- `force_download` (optional): Whether to re-download existing files
- `description` (optional): Human-readable description

### Pattern Examples

Common file patterns for different model types:

- **GGUF Models**: `["*Q4_K_M*", "*Q6_K*", "*Q8_0*"]`
- **PyTorch Models**: `["*.bin", "*.pt", "*.safetensors"]`
- **Config Files**: `["*.json", "*.txt", "config.yaml"]`
- **Exclude Documentation**: `["*.md", "*.gitattributes", "README*"]`

## Directory Structure

By default, models are organized as:

```
models/
├── microsoft/
│   └── DialoGPT-medium/
├── qwen/
│   ├── Qwen3-32B-GGUF/
│   └── Qwen3-30B-A3B-Instruct-2507-GGUF/
└── ggml-org/
    └── gpt-oss-120b-GGUF/
```

## Examples

### Download Specific Quantizations

```bash
# Download only Q8 quantized models
python download_hf_model.py --repo-id unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --allow-patterns "*Q8*"

# Download multiple quantization levels
python download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q4_K_M*" "*Q6_K*"
```

### Exclude Large Files

```bash
# Skip documentation and large unquantized files
python download_hf_model.py --repo-id microsoft/DialoGPT-medium --ignore-patterns "*.md" "*F32*" "*F16*"
```

### Custom Organization

```bash
# Download to specific directory structure
python download_hf_model.py --repo-id microsoft/DialoGPT-medium --local-dir ./conversations/dialog-medium
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `huggingface_hub` and `hf_transfer` are installed
2. **Permission Errors**: Check write permissions for the target directory
3. **Network Issues**: Verify internet connection and HuggingFace Hub access
4. **Disk Space**: Ensure sufficient disk space for large models

### Environment Variables

- `HF_HUB_ENABLE_HF_TRANSFER=1`: Enables faster downloads (set automatically)
- `HF_TOKEN`: Your HuggingFace token for private repositories

### Debugging

Add verbose output by checking the console messages. The script provides detailed feedback about:
- Download progress
- Pattern matching
- Error conditions
- Final file locations

## Migration from Static Script

If you're migrating from the old static script, you can:

1. Create a config file with your existing models:
```json
{
  "models": [
    {
      "repo_id": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "local_dir": "/home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "allow_patterns": ["*Q8*"]
    }
  ]
}
```

2. Run with the config:
```bash
python download_hf_model.py --config your_models.json
```

This provides the same functionality with much more flexibility for future use.

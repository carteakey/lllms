# /publish-model-post

Prepare a model-specific blog post for publication by converting it to be "L3MS-agnostic", saving it to the `carteakey.dev` blog repository, and symlinking it back to the `l3ms/docs/` directory.

## Usage

```
/publish-model-post <model-key> [--date <YYYY-MM-DD>]
```

**Arguments:**
- `<model-key>` — short slug used in file names (e.g. `qwen3-6-35b-a3b`, `gemma-4-26b-a4b`)
- `--date <YYYY-MM-DD>` — publication date for the blog post (defaults to today's date)

## Workflow

### Step 1 — Gather context
1. Read the existing draft at `docs/<model-key>-post.md` (if it exists) to capture benchmark outcomes, vision notes, and text narrative.
2. Read the `llama-swap.yaml` entry for this model to understand its precise parameters.
3. Read the `bench-models/run-llama-cpp-<model-key>.sh` (and vision variant if applicable) to see the exact `llama-server` CLI flags used.

### Step 2 — Draft the agnostic post in carteakey.dev
Create a new file at `/home/kchauhan/repos/carteakey.dev/src/posts/<YYYY-MM-DD>-running-<model-key>-locally.md`.

**The post must follow this specific "L3MS-agnostic" structure:**
- **Frontmatter**:
  ```yaml
  ---
  title: Running <Model Name> locally on <Hardware Profile>
  description: <Short description>
  date: <YYYY-MM-DD>
  updated: <YYYY-MM-DD>
  authored_by: ai-assisted
  draft: true
  tags:
    - AI
    - Self-Host
  pinned: false
  ---
  ```
- **TL;DR**: Bullet points for Model Repo, Stack (mainline `llama.cpp`), Benchmark outcomes (fit winner), and Vision/Memory-safe defaults.
- **End-to-end setup**:
  - **1) Build mainline llama.cpp**: Show standard `git clone` and `cmake` commands with standard `GGML_CUDA` flags.
  - **2) Download model**: Show the explicit `huggingface-cli download` command with the exact `--include` globs for the GGUF and (if applicable) mmproj.
  - **3) Run text server**: Show a fully expanded `llama-server \` command using the parameters extracted in Step 1.
  - **4) Run vision server** *(if applicable)*: Show the fully expanded `llama-server \` command including `--mmproj`, increased `fit-target`, and reduced batch sizes.
- **L3MS Callout**: Include this exact block after the setup instructions:
  ```markdown
  > **Easier path**: [carteakey/l3ms](https://github.com/carteakey/l3ms) wraps all of the above as pre-configured shell scripts along with a build helper, a model downloader, and bench scripts. Everything is editable text, not a UI form.
  ```
- **Benchmarks**: Include the markdown table showing the `pp` and `tg` numbers for the various strategies (baseline, fit, etc.) and state the hardware used (e.g. RTX 4070 12 GB).
- **Notes**: Include any model-specific gotchas, VRAM scaling issues, or stability notes.

### Step 3 — Set up the symlink
In the `l3ms` repository, replace the standalone markdown file with a symlink pointing to the `carteakey.dev` file.

```bash
cd ~/repos/l3ms
# Remove the old file if it isn't a symlink already
rm -f docs/<model-key>-post.md
# Create the symlink
ln -s /home/kchauhan/repos/carteakey.dev/src/posts/<YYYY-MM-DD>-running-<model-key>-locally.md docs/<model-key>-post.md
```

### Step 4 — Verify
Verify the symlink points to a valid file:
```bash
ls -la docs/<model-key>-post.md
cat docs/<model-key>-post.md | head -n 10
```

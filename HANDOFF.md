# L3MS Media Runtime Handoff

Date: 2026-08-15  
Deployment target: `yeti-cachy` (`ssh kpc-cachy-ts`)  
Live checkout: `/home/kchauhan/repos/l3ms`  
Baseline before this handoff commit: `065c58a0edd40d18e60939ef486ec58a0b87d913`

## Current state

The Rust L3MS media-runtime work is deployed and healthy on Yeti. The
MiniMax H3 local runtime is installed and has produced a playable MP4 on the
RTX 4070. LTX-2.5 and MiniMax Music are wired into the same declarative
workflow, but their credentials or model access are not complete:

- H3: ready for local audio/video generation.
- LTX-2.5: setup is ready, but Hugging Face reports that the gated model
  repository requires account approval. No LTX weights are installed.
- MiniMax Music: the `mmx` CLI is installed, but the private Music API key has
  not been entered on Yeti.
- `setup-media-runtimes.sh check` therefore exits non-zero until those two
  external prerequisites are resolved.

No secrets are stored in this repository or in this handoff.

## Target access and service

```sh
ssh kpc-cachy-ts
cd /home/kchauhan/repos/l3ms
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```

The user-level `llama-swap.service` is active after deployment. Its
authenticated API is on `127.0.0.1:8080`; the current Tailscale address is
`100.110.126.24`, so the reachable base URL is:

```text
http://100.110.126.24:8080/v1
```

The live checkout intentionally retains these host-local overrides and they
must not be overwritten by a future clean-clone deployment:

```text
M llama-swap.yaml
M model_downloader/models_config.json
```

The latest deployment backup was retained at:

```text
/home/kchauhan/repos/l3ms-backup-065c58a-20260815-160104
```

## First checks

```sh
./maintenance/setup-media-runtimes.sh check
./maintenance/run-l3ms-kpc.sh --list media
./target/release/l3ms --version
```

The check should report OK for the NVIDIA GPU, audio.cpp, all four H3
artifacts, ffmpeg/jq/xxd, the LTX checkout, Hugging Face auth material, and
the `mmx` executable. It will continue to warn about the five gated LTX
weights and Music authentication until those are fixed.

## H3 quick starts

Unload GPU-resident llama-swap models before a local H3 generation on this
12 GB card. Restart the service afterward if it was stopped for a smoke test.

```sh
./maintenance/run-l3ms-kpc.sh --media minimax-h3 \
  --extra '--prompt "A calm cinematic instrumental introduction"'

./maintenance/run-l3ms-kpc.sh --media minimax-h3 \
  --extra '--prompt-file ./prompt.txt --video --video-output ./clip.mp4'
```

The H3 wrapper preserves the raw RGB24 artifact and can mux a playable MP4
with ffmpeg, jq, and xxd. It uses CUDA, Q4_K GGUF weights, staged/layerwise
loading, and memory-saving defaults. The verified default video smoke was
`832x480`, `121` frames, and one diffusion step.

## Resolve the remaining blockers

### LTX-2.5

The Hugging Face token is present, but the account has not been approved for
the gated `Lightricks/LTX-2.5` repository. Accept the model terms in the
Hugging Face UI with the account that owns the token, then run on Yeti:

```sh
LTX_DOWNLOAD=1 ./maintenance/setup-media-runtimes.sh install-ltx
./maintenance/setup-media-runtimes.sh check
```

The expected official split files are listed by the check command. Do not
paste a token into chat or into this file.

### MiniMax Music

Enter the Music key directly on Yeti, without echoing it:

```sh
mmx auth login --api-key '<key entered locally>'
./maintenance/setup-media-runtimes.sh check
./maintenance/run-l3ms-kpc.sh --media minimax-music \
  --extra '--prompt "A short warm piano theme" --instrumental'
```

The wrapper supports `--prompt-file`, `--lyrics-file`, `--instrumental`, and
the hosted `mmx music generate` command. The key remains in the CLI's own
credential store.

## LTX variant decision

The configured LTX profile downloads the official BF16 component files and
uses `--quantization fp8-cast` with CPU/disk offload. This is the practical
Ada/RTX 4070 path documented by upstream. A direct pre-quantized NVFP4
download was not selected: upstream documents NVFP4 casting for Blackwell
(`SM >= 10`), while Yeti is Ada (`SM 8.9`). The official INT8 ConvRot option
is a ComfyUI-oriented path with a 32 GB+ VRAM expectation, also unsuitable
for this target. Revisit only if upstream publishes an Ada-compatible,
supported artifact and the runtime integration is verified.

## Files and source of truth

- `media-runtimes.json`: runtime/profile manifest.
- `src/media.rs`, `src/cli.rs`: profile discovery and safe argument boundary.
- `media-models/generate-minimax-h3.sh`: local H3 generation and MP4 mux.
- `media-models/generate-ltx-2.5.sh`: gated LTX generation, image inputs,
  prompt files, and offload/quantization options.
- `media-models/generate-minimax-music.sh`: hosted Music generation.
- `maintenance/setup-media-runtimes.sh`: readiness and installation checks.
- `maintenance/run-l3ms-kpc.sh`: Yeti launcher with noninteractive PATH setup.
- `docs/media-generation-runbook.md`: operator runbook and upstream links.
- `DETAILED_SESSION.md`: full investigation, verification, and deployment log.

## Verification already completed

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features --locked -- -D warnings`
- `cargo test --all-targets --all-features --locked`
- `cargo build --release --all-features --locked`
- Python compile checks and 14 downloader unit tests
- CLI version, quickstart, bench listing, media listing, and H3 help smoke
- Authenticated llama-swap `/v1/models`: HTTP 200, 13 models
- H3 default video: playable H.264/AAC MP4, 5.166667 seconds

## Safe next handoff

1. Resolve Hugging Face approval and install the five LTX files.
2. Authenticate `mmx` locally with the user's Music key.
3. Re-run `setup-media-runtimes.sh check` and one small generation per newly
   enabled profile.
4. Keep the two Yeti host-local override files during any future deployment.
5. Restart and health-check `llama-swap.service` after media tests that unload
   it.


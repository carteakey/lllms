# Media generation runbook

L3MS exposes media generation as an explicit, script-first CLI surface. The
registry in [`media-runtimes.json`](../media-runtimes.json) is the discoverable
source of truth; the executable wrappers in `media-models/` keep each upstream
runtime's flags and credentials separate from the Rust launcher.

## Target profile: Yeti/Cachy

Yeti has an RTX 4070 with 12 GiB of VRAM, 64 GiB of RAM, and CUDA. The selected
profiles therefore use the smallest practical local variants:

- **MiniMax-H3:** audio.cpp `release-0.6`, the normal `minimax_h3_q4_k` GGUF
  package, staged/layerwise weights, memory saver, and a 32×32 audio-first
  default. `--video` switches to 832×480 and 121 frames (about five seconds at
  24 fps). The optional INT8 ConvRot DiT is not the default because the
  audio.cpp performance report measured a higher peak-memory path for it.
- **LTX-2.5:** the official 22B distilled BF16 transformer with `fp8-cast` and
  CPU offload. The NVFP4 checkpoint is intended for newer hardware; the
  convolutional video VAE avoids requiring the diffusion-VAE attention extra.
- **MiniMax Music:** the official hosted Music Generation API through `mmx`.
  MiniMax-Music3 is not part of audio.cpp and its local checkpoint is not a
  sensible 12 GiB deployment, so the API profile is the supported standard
  music path.

These are conservative starting points, not a claim that every route will fit
or be fast on a 12 GiB card. Keep llama-swap's GPU model unloaded while running
H3 or LTX and start with five-second clips.

## Install and inspect

On Yeti, from the deployed L3MS checkout:

```bash
./maintenance/setup-media-runtimes.sh install-audio-cpp
./maintenance/setup-media-runtimes.sh install-ltx
./maintenance/setup-media-runtimes.sh install-music-cli
./maintenance/setup-media-runtimes.sh check
```

`install-audio-cpp` clones the pinned `release-0.6` branch, builds the CUDA
CLI/server for `minimax_h3`, and installs the public `minimax_h3_q4_k` package
under `${L3MS_MEDIA_ROOT:-$HOME/models/media}`. It does not touch llama-swap or
port 8080. On CachyOS it discovers `/opt/cuda` even from a non-interactive SSH
shell and pins the compatible CCCL 3.2 fetch needed by audio.cpp release-0.6.

LTX-2.5 is a gated Hugging Face model. Accept the model terms and log in with a
Read-scoped token, then opt into the roughly 66 GiB download:

```bash
export HF_TOKEN='…'        # kept in the shell environment, never committed
LTX_DOWNLOAD=1 ./maintenance/setup-media-runtimes.sh install-ltx
```

The script downloads only the files needed by the distilled wrapper: the
transformer, Gemma 4 projection encoder, convolutional video VAE, audio VAE,
and spatial upscaler. If authentication is not ready, `install-ltx` still sets
up the checkout and reports the missing gated files.

The official repository also publishes smaller-looking INT8 ConvRot and NVFP4
transformers, but they are not drop-in replacements here: INT8 ConvRot is for
ComfyUI rather than this PyTorch `ltx-pipelines` path, while the NVFP4 path
requires a Blackwell GPU (SM >= 10). Yeti's RTX 4070 is Ada (SM 8.9), so the
BF16 transformer with `fp8-cast` and CPU offload is the compatible quantized
runtime choice for this host.

Authenticate MiniMax Music separately. The wrapper does not accept or print a
key:

```bash
npm install --global mmx-cli
mmx auth login --api-key '<key>'
```

## CLI usage

List the profiles without contacting llama-swap:

```bash
cargo run --locked -- --list media
```

Run a profile interactively, optionally filtering the picker:

```bash
cargo run --locked -- --media h3
cargo run --locked -- --media ltx
cargo run --locked -- --media music
```

The wrappers use argv-safe `--extra` values. Prompts, lyrics, and local file
paths remain one argument and are never evaluated as shell code.

MiniMax-H3 audio-first music/audio:

```bash
cargo run --locked -- --media minimax-h3 --extra \
  '--prompt "slow modular synth arpeggio, warm tape saturation" --steps 20'
```

MiniMax-H3 short video with synchronized audio:

```bash
cargo run --locked -- --media minimax-h3 --extra \
  '--prompt "a rainy neon street, locked camera, reflections in puddles" --video'
```

LTX-2.5 text-to-video/audio:

```bash
cargo run --locked -- --media ltx-2.5 --extra \
  '--prompt "a paper boat crosses a puddle as thunder rolls in" --frames 121'
```

LTX-2.5 image-conditioned video (the image is passed as data, not shell code):

```bash
cargo run --locked -- --media ltx-2.5 --extra \
  '--prompt "the camera slowly pushes toward the boat" --image ./boat.png 0 0.8 --frames 121'
```

MiniMax hosted music:

```bash
cargo run --locked -- --media minimax-music --extra \
  '--prompt "dreamy ambient electronica with a gentle pulse" --instrumental'
cargo run --locked -- --media minimax-music --extra \
  '--prompt "upbeat indie pop" --lyrics-file ./lyrics.txt --lyrics-optimizer'
```

The default output directory is `$HOME/media-output`; set
`L3MS_MEDIA_OUTPUT_DIR` or pass `--output` to choose another destination.

## Inputs and computer-assisted workflows

The profiles expose text and lyrics as stable inputs, plus still-image
conditioning for the LTX DistilledPipeline. H3's official model family is
multimodal, but the local audio.cpp `FL2VA` package currently exposes the
text-to-audio/video route only. LTX image paths use explicit
`--image PATH FRAME STRENGTH` triples; audio-to-video and video-to-video work
still require their corresponding upstream LTX pipeline and model assets.

This keeps computer-use workflows safe: a UI or notebook can write a prompt or
lyrics file and invoke `l3ms --media … --extra …`, while local paths are passed
as data. No browser automation or API key is required by L3MS itself.

## Upstream references and research

- [MiniMax-H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3) —
  local H3-Base variants, multimodal capabilities, 768p local limit, and the
  recommended SGLang/vLLM/diffusers/ComfyUI runtimes.
- [audio.cpp release-0.6](https://github.com/0xShug0/audio.cpp/tree/release-0.6)
  and its [MiniMax-H3 community model guide](https://github.com/0xShug0/audio.cpp/blob/release-0.6/docs/community_models/minimax_h3.md)
  — GGUF package layout, CLI options, and staged/layerwise memory controls.
- [LTX-2.5 official quick start](https://github.com/Lightricks/LTX-2#-quick-start)
  — gated components, distilled pipeline, image conditioning, and
  `fp8-cast`/offload guidance.
- [MiniMax official CLI](https://github.com/MiniMax-AI/cli) and [Music API](https://platform.minimax.io/docs/api-reference/music-generation)
  — hosted music generation and authentication.
- Reddit reports from [the 12 GB DynamicVRAM thread](https://www.reddit.com/r/StableDiffusion/comments/1vghw05/minimax_h3_on_a_12gb_card_runaway_perstep/),
  [the H3 tips thread](https://www.reddit.com/r/StableDiffusion/comments/1vegtac/minimax_h3_tips_and_tricks_and_what_i_experienced/),
  and [the 16 GB timing thread](https://www.reddit.com/r/StableDiffusion/comments/1vjztod/26_sec_videos_on_16_gb_vram_rtx_5070_ti_and_only/) corroborate the
  practical advice to start at low resolution with substantial system RAM and
  expect multi-minute generation. Reddit is treated as field evidence, not as
  the source of runtime flags or model compatibility claims.

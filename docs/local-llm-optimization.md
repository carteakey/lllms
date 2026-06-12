---
title: "Local LLM Inference Optimization: The Complete Guide"
description: A living deep-dive into every layer of local LLM inference — hardware, OS, backend choice, and every llama.cpp flag worth tuning — synthesized from months of running large models on consumer hardware.
date: 2026-04-04
updated: 2026-04-04
authored_by: ai-assisted
draft: true
tags:
  - AI
  - Self-Host
  - llama.cpp
pinned: true
---

> **Note:** This post was drafted with significant AI assistance, synthesizing notes, bench results, and scripts from the [l3ms homelab toolkit](https://github.com/carteakey/l3ms) and the series of model-running posts on this site. The experiments, numbers, and failure modes documented here are real — the synthesis and prose are AI-assisted.

## Preface

Over the past year I've written posts on running [gpt-oss-120b](/blog/optimizing-gpt-oss-120b-local-inference/), [Qwen3-Coder-Next](/blog/qwen3-coder-next-40tps/), [Gemma 4 26B](/blog/gemma4-local/), and [Sarvam 30B](/blog/sarvam-local/) locally on consumer hardware. Each post has its own notes, failure modes, and tuning results — but the same lessons keep appearing: enable XMP, pin to P-cores, quantize your KV cache, don't trust the power profile.

This is my attempt at a master reference. Instead of re-discovering flags in every new model post, I want one doc to link back to. If you're hitting a performance wall, starting from scratch, or just want to understand what each knob actually does — start here.

The scope is intentionally wide. We start from "should I even run locally?" and drill all the way down to CUDA environment variables and specific failure modes. Skip to wherever you're stuck.

---

## 1. Glossary

| Term | Definition |
| --- | --- |
| **GGUF** | File format for quantized LLM weights used by llama.cpp. Stores weights, metadata, and tokenizer in a single binary. |
| **Quantization** | Reducing weight numerical precision (FP16 → Q4, etc.) to shrink model size and accelerate compute. More bits = higher quality, larger file. |
| **PP / Prompt Processing** | Tokens per second during the prefill phase — how fast the model reads your input. GPU-bound. |
| **TG / Token Generation** | Tokens per second during autoregressive decode — how fast you see output stream. Memory-bandwidth-bound. **This is what the user feels.** |
| **KV Cache** | Buffer storing attention Key/Value tensors for all prior context tokens. Grows linearly with context length. Lives in VRAM. |
| **Context Window** | Maximum total tokens (input + output) in one session. Determines KV cache size. |
| **Dense Model** | Standard transformer: all parameters active per token. Must fit in VRAM for full-speed inference. |
| **MoE / Mixture of Experts** | Architecture where each token activates only a small subset of "expert" networks. Enables very large total parameters with low per-token compute. Expert weights that don't fit in VRAM can spill to system RAM. |
| **Active Parameters** | For MoE: the subset of params computed per token. gpt-oss-120b: ~5B active of 120B total. Qwen3-Coder-Next: ~3B active of 80B total. TG speed tracks active count, not total. |
| **VRAM** | Video RAM — on-die GPU memory. Lowest latency, highest bandwidth storage for inference. |
| **Perplexity** | Statistical measure of model surprise on a test corpus. Lower = better. Standard proxy for comparing quantization quality levels. |
| **llama-bench** | CLI tool for synthetic PP and TG benchmarking, included in llama.cpp. |
| **llama-fit-params** | CLI tool that probes free VRAM and outputs optimal `-ngl` and `--override-tensor` flags without starting a server. |
| **`-ngl` / n-gpu-layers** | Number of transformer blocks to load onto GPU VRAM. |
| **`-ot` / override-tensor** | Regex-based per-tensor placement override — route specific weight tensors to CPU or GPU. |
| **XMP / EXPO** | BIOS profiles (Intel XMP / AMD EXPO) that run system RAM at its rated speed. "Auto" in BIOS often defaults to JEDEC base — a fraction of rated speed. |
| **Fit** | `--fit on` flag: auto-probes VRAM and computes optimal placement at startup, accounting for KV cache. |

---

## 2. The Inference Landscape

### 2.1 Why Run Locally?

- **Privacy**: prompts never leave your machine.
- **Cost**: zero marginal cost after hardware. Amortizes quickly under heavy use.
- **Control**: any model, any quant, any parameters. No deprecations, no rate limits, no pricing changes.
- **Offline**: works without internet.
- **Experimentation**: swap models, tune parameters, run evals without API contracts.

### 2.2 Cloud vs Local — Honest Tradeoffs

| | Hosted API | Self-hosted cloud GPU | Local hardware |
| --- | --- | --- | --- |
| Setup | Minutes | Hours | Hours–days |
| Model quality ceiling | Frontier | Your choice | Your choice |
| Per-token cost | $1–15/M | $0.10–0.50/GPU-hr | Zero (marginal) |
| Privacy | Provider sees data | You control | Full |
| Hardware investment | None | None | $500–$3000+ |

Most serious users end up with both: cloud APIs for frontier tasks, local for everything privacy-sensitive, experimental, or routine.

### 2.3 Local Inference Tools

| Tool | Best for | Notes |
| --- | --- | --- |
| **llama.cpp** | Performance tuning, full flag control, any hardware | Build from source; CLI-centric |
| **Ollama** | Zero-config, model management, Docker | Uses llama.cpp internally; limited tuning surface |
| **LM Studio** | Desktop GUI, Windows/macOS, model browsing | Good for evaluation; minimal performance control |
| **vLLM** | Multi-user production serving, continuous batching | Designed for full-VRAM serving; not suited for consumer hybrid setups |
| **exllamav2** | Maximum speed for dense models | CUDA-only; excellent for models that fully fit in VRAM |
| **mlx** | Apple Silicon | macOS only; leverages unified memory; no CUDA |

**This guide focuses on llama.cpp.** Most concepts (KV cache, quantization, layer placement) generalize across tools.

### 2.4 Backends Within llama.cpp

| Backend | Build flag | Best for |
| --- | --- | --- |
| **CUDA** | `-DGGML_CUDA=ON` | NVIDIA GPUs; highest performance and most tested |
| **Vulkan** | `-DGGML_VULKAN=ON` | AMD and Intel GPUs; cross-platform; works on NVIDIA too |
| **Metal** | (auto on macOS) | Apple Silicon; unified memory |
| **CPU-only** | (no GPU flag) | Reference; small models or debugging |
| **RPC** | `-DGGML_RPC=ON` | Experimental distributed inference |

> **This document assumes CUDA.** Where behavior differs on Vulkan or CPU-only, it's marked `[Vulkan]` or `[CPU]`. If you're on AMD/Intel GPU, most flag logic is the same but CUDA-specific env vars don't apply.

---

## 3. Hardware

### 3.1 The Memory Hierarchy — The Most Important Mental Model

Token generation speed is limited by how fast you can read model weights from storage. Rough bandwidth numbers:

```
VRAM (GPU on-die)              ~600–1000 GB/s
Unified memory (Apple Silicon)  ~200–400 GB/s
System RAM                       ~50–200 GB/s   (varies hugely by speed and channel config)
NVMe SSD                          ~5–7 GB/s
SATA SSD / HDD                    ~0.5–3 GB/s
```

**Dense models**: every token reads all active weights. Model must fit in VRAM for full-speed inference. Any spill to system RAM causes a large TG drop.

**MoE models**: only a fraction of expert weights is needed per token, but those weights must still be streamed — often from system RAM. TG ∝ RAM bandwidth. This means a RAM kit running at rated speed vs JEDEC base can deliver 2–3× the bandwidth, translating almost linearly to TG throughput on MoE models.

**Check your RAM is at its rated speed:**
```bash
sudo dmidecode -t memory | grep -E "Speed|Configured"
# "Configured Memory Speed" must match your XMP/EXPO profile speed.
# If it doesn't, enable XMP/EXPO in BIOS.
```

This is consistently one of the largest single wins available for MoE inference. "Auto" BIOS settings commonly default to JEDEC base speed regardless of the kit's rating.

### 3.2 GPU / VRAM

VRAM is the primary inference resource. More VRAM = more layers on GPU = faster inference.

| VRAM | Dense examples | MoE hybrid examples |
| --- | --- | --- |
| 8 GB | 7B–13B Q4 | 30B–70B, heavy RAM offload |
| 12 GB | 13B–20B Q4; 7B Q8 | 70B–120B+ with CPU offload |
| 24 GB | 34B Q4; 13B Q8 | 120B+ moderate offload |
| 48 GB+ | 70B Q8; 34B FP16 | Most MoE partially/fully on GPU |
| 80 GB+ | 120B+ FP16 | No offload needed |

**iGPU display trick (desktop NVIDIA):** route display through the motherboard video output instead of the GPU. This frees 500–1000 MB VRAM the GPU was using for desktop composition.

### 3.3 CPU

For **dense models** (all in VRAM): CPU is nearly idle during inference. Core count has minimal impact.

For **MoE hybrid**: CPU executes expert forward passes for every token. Expert compute is the TG bottleneck. **Intel hybrid (12th gen+):** P-cores and E-cores have significantly different throughput for matrix operations. E-cores drag TG down by 20–30%. Always pin to P-cores:

```bash
taskset -c 0-11 llama-server ...   # i5-12600K: cores 0-11 are P-cores
```

Thread count: set `--threads` to P-core count, leaving 1–2 for the OS. More threads than P-cores is counterproductive.

### 3.4 What Is "Good Enough" Throughput?

| TG speed | Experience |
| --- | --- |
| < 5 t/s | Painful; barely usable |
| 5–10 t/s | Functional; near human reading speed (~7 t/s) |
| 10–20 t/s | Comfortable for interactive chat |
| 20–40 t/s | Fast; coding agents feel snappy |
| 40+ t/s | Near-instant for most tasks |

For coding agents: TG dominates. First-token latency matters less once the context is warm. Every t/s gained compounds across a full session.

---

## 4. OS Choice

### 4.1 Linux

Highest-performance path for CUDA inference.

- **~15–20% TPS advantage** over Windows in practice: leaner CUDA driver overhead, better scheduling under sustained load, direct memory control.
- Full access to CPU governor, NUMA, huge pages, cgroups, headless mode.
- Recommended distributions: CachyOS (real-time kernel, best power management tuning surface), Ubuntu/Debian (easiest CUDA packages), Arch (rolling, latest drivers).

### 4.2 Windows

Reasonable; CUDA support is solid.

- Some overhead from Windows scheduler and CUDA runtime.
- **Power plan**: set to "High Performance" or "Ultimate Performance". Default "Balanced" throttles CPU under sustained inference load.
- NVIDIA Control Panel → "Power management mode" → "Prefer maximum performance".
- WSL2: close to native for CUDA; some VRAM overhead from virtualization. Generally acceptable if dual-boot isn't an option.

### 4.3 macOS

Different runtime stack — CUDA guidance does not apply.

- Metal backend activates automatically.
- **Apple Silicon unified memory**: CPU and GPU share the same physical pool. A machine with 128 GB RAM effectively has 128 GB "VRAM". Transforms the dense model size ceiling.
- Memory bandwidth is excellent (~400 GB/s on M3 Max); competitive with mid-range NVIDIA for the models it can run.
- `mlx` framework is worth evaluating alongside llama.cpp for Metal workloads.

### 4.4 Linux OS Tuning

These settings have measurable impact on TG throughput.

#### CPU Governor and Power Profile
```bash
# Must be "performance" on all cores
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# EPP must also be "performance" — governor alone is not enough on intel_pstate
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference

# Verify actual P-core frequency near max boost
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -6
```

**The `power-profiles-daemon` trap**: KDE and GNOME ship with `power-profiles-daemon`, which can set a non-performance HWP (hardware P-state) mode on some boots. The insidious symptom: all sysfs checks (`scaling_governor`, `energy_performance_preference`, `cpu MHz`) report "performance" — but TG runs 20–30% below expected, and it varies between boots. The degradation happens at the hardware MSR level where standard tooling doesn't look.

Fix — replace with `tuned-ppd`:
```bash
# Arch / CachyOS
sudo pacman -S tuned-ppd       # removes power-profiles-daemon automatically
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
# Reboot. TG will be stable and correct across all boots.

# Ubuntu / Debian
sudo apt install tuned
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
```

#### Transparent Huge Pages
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Recommended: [always]
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

#### Headless Mode
```bash
# Stop desktop compositor — frees 200-400 MB RAM + compositor VRAM
sudo systemctl isolate multi-user.target
sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Restore when done
sudo systemctl isolate graphical.target
```

Without a display server, use `zellij` in a TTY for split panes: `zellij` in any TTY gives full terminal multiplexing without X or Wayland.

---

## 5. Why llama.cpp, and How to Build It

### 5.1 Why Not Ollama?

Ollama uses llama.cpp internally but exposes a minimal, fixed-default configuration surface. Every flag in §7–§14 of this guide — layer placement, KV quantization, fit parameters, batch sizes, CUDA env vars — is unavailable or unexposed in Ollama.

Ollama is the right choice for quick setup and model management. If you're reading this guide, you've outgrown it.

### 5.2 Building from Source

Always build from source. Distro packages are outdated and not compiled for your GPU. MoE inference performance improves significantly with each llama.cpp release.

**CUDA build:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
#  89 = RTX 40-series | 86 = RTX 30-series | 75 = RTX 20-series | 61 = GTX 10-series

cmake --build . --config Release \
  --target llama-server llama-bench llama-fit-params llama-cli --parallel
```

**Vulkan build** (AMD, Intel GPU):
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON
# CUDA-specific build flags do not apply here
```

Keep your build updated. Pull and rebuild regularly — especially before benchmarking a new model.

### 5.3 Key Binaries

| Binary | Purpose |
| --- | --- |
| `llama-server` | OpenAI-compatible HTTP inference server |
| `llama-bench` | Synthetic PP and TG benchmarking |
| `llama-fit-params` | Dry-run VRAM probe → outputs optimal `-ngl` + `-ot` flags |
| `llama-cli` | Interactive CLI prompt for quick tests |
| `llama-sweep-bench` | Sweep across parameters (batch size, ngl, etc.) |

---

## 6. Model Selection and Quantization

### 6.1 Dense vs MoE — Choose Your Tuning Strategy

| | Dense | MoE |
| --- | --- | --- |
| Active params/token | All | Small fraction (e.g. 3B of 80B) |
| Must fit in VRAM | Yes, for best performance | No — experts can live in RAM |
| Primary tuning lever | Quant level + context | Layer placement + RAM bandwidth |
| TG speed driver | VRAM bandwidth | RAM bandwidth + GPU layer count |
| Example models | Gemma 4, Llama 3, Mistral | Qwen3, DeepSeek, gpt-oss |

### 6.2 Quantization Reference

| Quant | Size vs FP16 | Quality | Notes |
| --- | --- | --- | --- |
| Q2_K | ~25% | Noticeable degradation | Use only under extreme size constraints |
| Q4_K_M | ~35% | Good | Best size/quality balance for most situations |
| Q4_K_XL / UD-Q4_K_XL | ~35% | Better than Q4_K_M | Unsloth Dynamic: layer-importance-aware allocation |
| Q5_K_M / Q5_K_XL | ~40% | Very close to FP16 | Strong default when VRAM allows |
| Q6_K | ~50% | Near-lossless | High-VRAM setups |
| Q8_0 | ~65% | Effectively lossless | If storage/RAM permits |
| FP16 | 100% | Reference | Maximum quality; maximum size |
| MXFP4 (native) | ~35% | Better than Q4_K_M | gpt-oss models: trained in MXFP4, not post-quantized |

**UD (Unsloth Dynamic) quants**: allocate higher bits to attention-sensitive layers and lower bits to robust layers. Better perplexity than uniform quants at the same average bit width. Generally the best choice when available.

**Rule of thumb**: use the highest quant that fits your VRAM + RAM budget. Q5_K_XL or UD-Q5_K_XL is a strong default. Drop to Q4 only when necessary.

---

## 7. Layer Placement — The Core Optimization for MoE

> For dense models fully in VRAM: use `-ngl 99` and skip to §8.

For MoE hybrid setups, layer placement is where most performance lives. The goal: keep as many blocks as possible on GPU (especially early layers and attention), while offloading expert weights to RAM.

### 7.1 `--n-gpu-layers` (`-ngl`)

How many transformer blocks to load onto GPU. Start at `99` (all layers). Drop if CUDA OOM.

```bash
llama-server -m model.gguf --n-gpu-layers 99    # all on GPU
llama-server -m model.gguf --n-gpu-layers 37    # 37 on GPU, rest on CPU
```

### 7.2 `--n-cpu-moe`

Integer count: keep the named number of MoE layers' expert weights on CPU. Quick coarse control.

```bash
--n-cpu-moe 31   # first 31 MoE layer experts on CPU
```

> ⚠️ **RAM ceiling**: for a ~60 GB model, putting all experts on CPU tries to load ~60 GB into RAM. On a 64 GB system this will hard-crash the machine. Always use `llama-fit-params` (§7.4) to find safe values before setting this high.

### 7.3 `--override-tensor` (`-ot`) — Fine-Grained Placement

Per-tensor, per-layer placement via regex. Most control, most complexity.

```bash
# All expert projections in all layers → CPU:
--override-tensor ".ffn_(up|down|gate)_(ch|)exps=CPU"

# Layers 5+ experts → CPU; keep layers 0-4 fully on GPU:
--override-tensor "blk\.(5|[6-9]|[0-9][0-9]+)\.ffn_(up|down|gate)_(ch|)exps=CPU"
```

**The shared expert gotcha**: some models (Qwen3.5-122B, certain gpt-oss variants) have two expert tensor families:
- Routed experts: `ffn_{up,down,gate}_exps`
- Shared expert (always active, 1 per layer): `ffn_{up,down,gate}_shexp`

A pattern matching only `_exps` leaves `_shexp` on GPU, silently consuming VRAM and causing CUDA OOM. The safe pattern captures both:

```bash
# (ch|) matches both _exps (routed) and _shexp (shared):
--override-tensor ".ffn_(up|down|gate)_(ch|)exps=CPU"
```

Safe to include `(ch|)` even for models without shared experts — it's harmless and future-proofs the pattern.

### 7.4 `--fit on` — Recommended Starting Point

Auto-probes free VRAM at startup, computes optimal `-ngl` + `-ot` placement automatically. Zero manual tuning.

```bash
llama-server \
  -m model.gguf \
  --fit on \
  --fit-ctx 65536 \   # minimum context to guarantee fits; KV cache for this ctx is accounted for
  --fit-target 512 \  # VRAM headroom in MiB to leave free
  ...
```

**`--fit-target` note**: CUDA's VMM pool grows in ~1 GiB chunks as context depth increases during a session. Using `--fit-target 128` produces great bench numbers but can OOM mid-session when the pool needs to grow. Use `≥512 MiB` for production text servers. Use `2048` for vision models where the mmproj allocation is large.

**Dry run without starting a server:**
```bash
llama-fit-params \
  -m /path/to/model.gguf \
  -fitt 512 \     # fit-target MiB
  -fitc 65536     # fit-ctx tokens
# Outputs something like: -c 65536 -ngl 49 -ot "blk\.8\.ffn_...=CPU,..."
```

This output is what you hardcode for static placement.

### 7.5 Static vs Dynamic Placement

| | `--fit on` | Hardcoded `-ngl` + `-ot` |
| --- | --- | --- |
| Startup | +1–5s probe delay | Instant |
| Placement | Fresh each boot | Fixed |
| Tuning effort | None | One-time from `llama-fit-params` |
| Best for | Experimentation | Production |
| Reproducibility | Good (if VRAM is free) | Deterministic |

For a stable homelab server, derive placement once with `llama-fit-params` and hardcode it in your run script. Use `--fit on` when testing new models or after hardware changes.

---

## 8. Context and KV Cache

### 8.1 `--ctx-size` — Choosing Context Length

The KV cache grows linearly with context and lives in VRAM. Large context on small VRAM can push expert layers off GPU.

Approximate KV VRAM usage (varies by model, head count, and layer structure):

| Context | f16 KV | q8_0 KV |
| --- | --- | --- |
| 8k | ~0.5 GB | ~0.25 GB |
| 32k | ~2 GB | ~1 GB |
| 64k | ~4 GB | ~2 GB |
| 128k | ~8 GB | ~4 GB |

On a 12 GB card at 128k context with f16 KV, 8 GB goes to KV cache — leaving only ~4 GB for model weights and attention. Context choice directly affects how many GPU layers you can afford.

**Practical guidance:** coding sessions work well at 64k; long-context RAG may need 128k+; vision inference is typically safest at 64k on 12 GB VRAM.

### 8.2 KV Cache Quantization (`-ctk`, `-ctv`) [CUDA]

Quantizing the KV cache halves (q8_0) or further reduces (q4_0) its VRAM footprint.

```bash
-ctk q8_0 -ctv q8_0
```

| KV quant | VRAM vs f16 | Quality | Recommendation |
| --- | --- | --- | --- |
| `f16` | 1× | Lossless | Default |
| **`q8_0`** | **0.5×** | **Effectively lossless** | **Default recommendation** |
| `q4_0` | ~0.33× | Some degradation at long context | Only under extreme VRAM pressure |

**The compounding effect**: on a 12 GB card with 64k context, switching f16 → q8_0 KV frees ~2 GB. That 2 GB lets `llama-fit-params` keep one to two additional GPU layers — translating directly to higher TG. Confirmed on Qwen3-Coder-Next: q8_0 KV at 64k unlocked 2 extra GPU layers and added ~2 t/s TG vs f16 KV.

> At short bench contexts (512 tokens), the KV cache is tiny and this effect is near-zero. Always test at your real serving context length.

### 8.3 `--parallel` — Concurrent Inference Slots

Each slot maintains its own KV cache. `--parallel 4` multiplies KV VRAM by 4.

```bash
--parallel 1   # single user homelab: reclaims KV VRAM for model weights
```

On gpt-oss-120b, dropping `--parallel 4` → `--parallel 1` freed ~540 MiB VRAM — enough for one more GPU layer and +1 t/s TG.

### 8.4 `--flash-attn on` (`-fa`)

Reduces attention memory from O(n²) to O(n log n). Required for large context on consumer hardware. No meaningful downside on CUDA.

```bash
--flash-attn on
```

Always enable. Required for certain KV quantization types on some configurations.

> [Vulkan]: Flash attention support varies by driver version. Verify before relying on it.

---

## 9. Batch Sizes

### 9.1 `--batch-size` (`-b`) — Prompt Processing Throughput

Controls how many tokens are processed in one forward pass during prefill. Higher = better PP throughput; more VRAM required.

```bash
--batch-size 2048   # high throughput (~maximum for most models)
--batch-size 1024   # balanced; good default
--batch-size 512    # conservative; use for vision or tight VRAM
```

Reduce if you hit CUDA OOM during the prefill phase specifically.

### 9.2 `--ubatch-size` (`-ub`) — Physical Micro-Batch

Physical sub-batch within a logical batch. Must be ≤ `--batch-size`.

```bash
--ubatch-size 512   # typical default
```

**Vision/multimodal critical**: an image tokenizes to several hundred tokens. If `--ubatch-size` < image token count, llama.cpp throws an assertion during vision inference. Use `--ubatch-size 512` or higher and test with your actual image sizes. On 12 GB VRAM, `--batch-size 256 --ubatch-size 512` is a stable vision baseline.

---

## 10. Sampling Parameters

Sampling controls the probability distribution at each decode step. These affect output quality and — via vocabulary truncation (`top-k`) — slightly affect speed.

**Start with model card defaults.** Most GGUF releases specify tested values. Use those before experimenting.

### 10.1 `--temp` — Temperature

- `0.0`: greedy / deterministic. Best for coding agents where reproducibility matters.
- `0.7`: standard creative chat.
- `1.0`: no rescaling; follows raw model distribution. Most modern instruction-tuned models are calibrated for this.

### 10.2 `--top-k` — Vocabulary Truncation

Keeps only top K most probable tokens before sampling.

- `0`: full vocabulary (default; most diverse; slowest to compute)
- `100`: safe performance cap — confirmed no measurable quality loss on coding tasks (gpt-oss-120b)
- `20–64`: model-specific tighter caps

### 10.3 `--top-p` — Nucleus Sampling

Filters to tokens whose cumulative probability ≥ p. Applied after `top-k`.

- `1.0`: no filtering (gpt-oss, Sarvam defaults)
- `0.95`: standard for chat/code (Gemma 4, Qwen3 defaults)

### 10.4 `--min-p`

Filters tokens below `min-p × max_token_probability`.

- `0.0` off (most models)
- `0.01` light floor (Qwen3-Coder-Next)

### 10.5 `--repeat-penalty`

- `1.0`: no penalty. Recommended for code — code naturally repeats patterns (variable names, keywords) and penalizing them degrades output.
- `1.1–1.3`: mild penalty for prose.

### Recommended Defaults by Model Family

| Model | temp | top-k | top-p | min-p | repeat-penalty |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 | 1.0 | 64 | 0.95 | 0.0 | 1.0 |
| Qwen3 / Qwen3-Coder | 1.0 | 40 | 0.95 | 0.01 | 1.0 |
| gpt-oss-120b | 1.0 | 100 | 1.0 | 0.0 | 1.0 |
| Sarvam | 1.0 | 20 | 1.0 | 0.0 | 1.0 |

---

## 11. Threading and CPU Control

### 11.1 `--threads` (`-t`)

CPU threads for the token generation phase (expert compute in hybrid MoE setups).

```bash
--threads 10   # P-core count minus 1-2 for OS headroom
```

More threads than available P-cores is counterproductive — they contend for the same memory bus and typically reduce TG.

### 11.2 `--threads-batch`

CPU threads for the PP (prefill) phase. PP is a burst workload; you can set this to full thread count.

```bash
--threads-batch 12
```

### 11.3 `taskset` — P-Core Pinning [Linux]

Most reliable way to keep inference off E-cores on Intel 12th gen+ (and other hybrid architectures):

```bash
taskset -c 0-11 llama-server ...   # pin all process threads to P-cores
```

Verify your P-core range from CPU documentation or `lstopo`. On Intel 12600K, cores 0–11 (6 P-cores × 2 threads) are the P-cores; 12–15 are E-cores.

### 11.4 `--poll`

Controls CPU spin aggressiveness while waiting for GPU kernel completion.

- `0`: yield/sleep
- `100`: busy spin

**On hybrid CPU+GPU inference, this is flat.** GPU kernel execution and PCIe transfer dominate synchronization. Confirmed across multiple sweeps — within noise at all poll levels. Leave at default `50` or set to `0` to reduce idle CPU load. Do not tune this.

### 11.5 `--numa`

NUMA affinity modes: `distribute`, `isolate`, `numactl`.

**Single-socket systems: skip.** There is only one NUMA node; these modes provide no benefit and can hurt performance. Use `taskset -c` for affinity instead.

Relevant only on dual-socket server hardware (AMD EPYC, Intel Xeon) where NUMA topology is real.

---

## 12. Memory Control

### 12.1 `--no-mmap`

Without this, llama.cpp uses memory-mapped I/O. Expert weight accesses during decode are non-sequential — the OS page fault handler triggers repeatedly for cold pages, adding latency jitter to TG.

With `--no-mmap`, the entire model loads into RAM before inference begins. No page faults.

```bash
--no-mmap   # recommended for all hybrid MoE and persistent server setups
```

Tradeoff: longer startup. Worth it for any persistent server.

### 12.2 `--mlock`

Pins model pages in RAM, preventing the OS from swapping them under memory pressure.

```bash
--mlock
```

Important when `vm.swappiness` is high (many Linux distributions default to 60–150 with ZRAM) or when running close to the RAM ceiling. Without it, a swap event mid-session can make TG appear to stall. Skip only if RAM is critically tight.

---

## 13. Priority and Process Settings

### 13.1 `--prio`

Scheduling priority for the inference process. Scale 0–3.

```bash
--prio 2   # high priority; reduces OS scheduling jitter on TG
```

### 13.2 `--no-warmup`

Skips initial kernel warmup pass at startup (compiles CUDA kernels on first real request instead).

```bash
--no-warmup   # reduces startup time; safe for persistent servers
```

---

## 14. CUDA-Specific Settings [CUDA]

### 14.1 Environment Variables

```bash
export LLAMA_SET_ROWS=1         # Improves CPU cache locality for MoE expert row access
export GGML_CUDA_GRAPH_OPT=1    # Batches CUDA kernel launches; reduces dispatch overhead
```

**`GGML_CUDA_GRAPH_OPT` caveat**: CUDA graph optimization captures the kernel graph at a specific context depth. When depth increases significantly (long agentic sessions), CUDA triggers a re-capture, growing the VMM pool. On tight-fit configs (< 512 MiB headroom), this causes mid-session OOM. If you observe intermittent CUDA OOM on long sessions, set `GGML_CUDA_GRAPH_OPT=0`.

### 14.2 Notable Build Flags

| Flag | Effect |
| --- | --- |
| `GGML_CUDA_GRAPHS=ON` | Enables CUDA graph capture at build time |
| `GGML_CUDA_F16=ON` | f16 CUDA ops |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Flash attention for all quant types |
| `GGML_NATIVE=ON` | CPU-native optimization; don't use for distributed binaries |
| `GGML_LTO=ON` | Link-time optimization; slower build, faster runtime |

### 14.3 cuBLAS — Tested and Closed

`GGML_CUDA_FORCE_CUBLAS=ON` forces CUDA BLAS routines over the default GGML MMQ (mixed-precision matrix quantization) kernels.

Tested on mxfp4 and Q4 models: **slower** than default. GGML MMQ has native mxfp4/Q4 paths tuned for consumer decode batch sizes (1–16 tokens). cuBLAS is optimized for large datacenter batches. Result: ~45 t/s PP regression, no TG improvement. Default build wins on consumer hardware. May be worth re-evaluating on 24+ GB cards where larger batch sizes make cuBLAS more competitive.

---

## 15. Vision / Multimodal

### 15.1 `--mmproj`

Path to the multimodal projector file:

```bash
--mmproj /path/to/mmproj-BF16.gguf
```

Typically 1–3 GB. Allocates in VRAM at startup alongside the model.

### 15.2 OOM Failure Modes on Constrained VRAM

**Failure 1 — mmproj allocation**: the projector needs contiguous VRAM at load time. If `--fit-target` left only a small margin, the allocation fails. Symptom: crash at model load (not during inference).

Fix: use `--fit-target 2048` for vision models.

**Failure 2 — batch assertion**: image token count exceeds `--ubatch-size`. An image can tokenize to several hundred tokens; if the batch is too small, llama.cpp asserts.

Fix: use `--ubatch-size 512` or higher.

### 15.3 Safe Vision Profile (12 GB VRAM)

```bash
llama-server \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --ctx-size 65536 \
  --fit on --fit-ctx 65536 --fit-target 2048 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --batch-size 256 --ubatch-size 512 \
  --no-mmap --mlock \
  --parallel 1
```

Separate text and vision servers on different ports if running both workloads from the same GPU.

---

## 16. ik_llama.cpp Fork [Advanced]

[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is a fork with MoE-specific kernel optimizations not yet upstream. Worth evaluating once you've hit the ceiling on stock llama.cpp.

### 16.1 Key Flags

| Flag | Effect | Cost |
| --- | --- | --- |
| `-fmoe` (fused-moe; **default on**) | Fused MoE expert kernel: significant TG uplift | PP drops ~50% |
| `-muge` (merge-up-gate) | Repacks up+gate projections; better read locality | +25–30 GB RAM; slow startup |
| `-mqkv` (merge-qkv) | Merges Q, K, V projections | Small TG gain; no RAM cost |
| `-ger` (grouped-expert-routing) | Groups token-expert assignments for cache locality | Variable; sweep to confirm |

### 16.2 Real Numbers (Qwen3-Coder-Next, RTX 4070 12 GB)

| Config | pp512 (t/s) | tg128 (t/s) | Notes |
| --- | ---: | ---: | --- |
| Upstream baseline (N_CPU_MOE=40) | **453** | 40.6 | Reference |
| ik fused-moe (default on) | 215 | 40.5 | PP halves; TG same as upstream |
| ik fused-moe + merge-qkv | 217 | 40.9 | Marginal gain over fused alone |
| ik fused-moe + merge-up-gate | 215 | **41.1** | Best TG; +27 GB RAM overhead |
| ik fused-moe + gcr + merge-up-gate | 215 | 41.2 | Diminishing returns |

On gpt-oss-120b, ik_llama was also tested: CUDA graph compilation overhead dominated; PP collapsed to ~98 t/s (vs 428 upstream) and TG dropped to 20 t/s (vs 28 upstream). Not competitive for that model. Performance is model-architecture-dependent.

### 16.3 When to Use

- **Use ik_llama** if TG is your sole bottleneck, you have RAM headroom, and PP regression is acceptable (generation-heavy agent loops)
- **Stay on upstream** for RAG, long-prompt workloads, or when PP matters. Upstream also has better startup predictability and no extra RAM cost.

---

## 17. Diagnostic Checklist

Run before benchmarking or when TG is unexpectedly low.

```bash
# 1. RAM speed — most common culprit for MoE TG underperformance
sudo dmidecode -t memory | grep -E "Speed|Configured"
# "Configured Memory Speed" must match your XMP/EXPO rated speed

# 2. CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance

# 3. EPP — must be "performance" (governor alone is not enough on intel_pstate)
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
# Expected: performance (not balance_performance, not powersave)

# 4. Actual CPU frequency — P-cores should be near rated max boost
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -6

# 5. Free VRAM — start inference from near-empty
nvidia-smi | grep MiB

# 6. Thermal — sustained load under throttle temperature
cat /sys/class/thermal/thermal_zone*/temp
# In millidegrees; 80000 = 80°C; throttle typically starts 85–105°C depending on chip

# 7. Background CPU hogs
ps aux --sort=-%cpu | head -10

# 8. Swap activity (high vm.swappiness systems)
cat /proc/vmstat | grep -E "pswpin|pswpout"
# Growing non-zero values = model weights being swapped mid-session

# 9. PCIe link speed [CUDA]
nvidia-smi -q | grep -A 3 "PCIe Generation"
# Expected: Current Gen = 3 or 4

# 10. Active tuned profile (if using tuned-ppd)
sudo tuned-adm active
# Expected: throughput-performance
```

### Known TG Variability Root Causes

| Root cause | Symptom | Fix |
| --- | --- | --- |
| `power-profiles-daemon` degrading HWP | TG varies 20–30% between boots; all sysfs checks look fine | Replace with `tuned-ppd` + `throughput-performance` |
| RAM not at rated speed (XMP/EXPO off) | TG 2–3× below expected; stable but low | Enable XMP/EXPO in BIOS |
| E-cores included in thread range | TG lower than P-core-only baseline | `taskset -c <p-cores-only>` |
| `vm.swappiness` + model too large for RAM | TG stalls mid-session | `--mlock`; reduce model or add RAM |
| `GGML_CUDA_GRAPH_OPT=1` with varying context | Intermittent OOM at long prompts | Set `GGML_CUDA_GRAPH_OPT=0` |
| `--fit-target` too small | OOM mid-session (not at startup) | Increase `--fit-target` to ≥512 MiB |
| Vision `--fit-target` too small | Crash at model load with mmproj | Use `--fit-target 2048` for vision |

---

## 18. Optimization Priority Checklist

Ordered by typical impact. Each item one line; link to section for detail.

| # | Action | Impact | Section |
| --- | --- | --- | --- |
| 1 | **Enable XMP/EXPO in BIOS** | 2–3× TG on MoE | §3.1 |
| 2 | **Run Linux** or tune Windows power plan | ~15–20% TPS | §4 |
| 3 | **Replace `power-profiles-daemon` with `tuned-ppd`** | Eliminates intermittent 20–30% TG drop | §4.4 |
| 4 | **Build llama.cpp from source; keep updated** | MoE kernel improvements per release | §5.2 |
| 5 | **Use `--fit on`** for VRAM-optimal layer placement | Major TG; no manual tuning | §7.4 |
| 6 | **Use `-ctk q8_0 -ctv q8_0`** | Frees KV VRAM → extra GPU layers → +TG | §8.2 |
| 7 | **Set `--parallel 1`** for single-user homelab | Reclaims KV VRAM for weights | §8.3 |
| 8 | **Pin to P-cores** with `taskset -c` | +20–30% TG on Intel hybrid | §11.3 |
| 9 | **Enable `--flash-attn on`** | Required for large-context stability | §8.4 |
| 10 | **Enable `--no-mmap`** | Eliminates TG jitter from page faults | §12.1 |
| 11 | **Enable `--mlock`** | Prevents mid-session swap degradation | §12.2 |
| 12 | **Go headless** (`systemctl isolate multi-user.target`) | Frees 200–400 MB RAM + compositor VRAM | §4.4 |
| 13 | **iGPU for display** (motherboard HDMI) | Frees 500–1000 MB VRAM | §3.2 |
| 14 | **Set `LLAMA_SET_ROWS=1`** | Cache locality for MoE expert access | §14.1 |
| 15 | **Set `GGML_CUDA_GRAPH_OPT=1`** (fixed-depth only) | Reduces CUDA dispatch overhead | §14.1 |
| 16 | **Evaluate ik_llama.cpp** (generation-heavy) | +TG at cost of PP | §16 |

---

## Changelog

| Date | Note |
| --- | --- |
| 2026-04-04 | Initial post — synthesized from l3ms scripts, bench-runbook, and model posts. |

## References

- [l3ms — homelab LLM toolkit (scripts, bench runbook, run scripts)](https://github.com/carteakey/l3ms)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [ik_llama.cpp fork](https://github.com/ikawrakow/ik_llama.cpp)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [gpt-oss-120b optimization post](/blog/optimizing-gpt-oss-120b-local-inference/)
- [Qwen3-Coder-Next 40 t/s post](/blog/qwen3-coder-next-40tps/)
- [Gemma 4 26B local post](/blog/gemma4-local/)
- [Sarvam 30B local post](/blog/sarvam-local/)

# Local Qwen3‑Coder Environment

This project provides a pair of PowerShell scripts that let you download, build and run the [**Qwen3‑Coder‑30B‑A3B‑Instruct‑1M**](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF) model locally on Windows via **\[`ik_llama.cpp`]** – a performance‑oriented fork of `llama.cpp`.

The workflow is self‑contained:

```
repo/                     # your checkout
├─ vendor/                # `ik_llama.cpp` source cloned & built here
└─ models/                # downloaded GGUF model(s)
```

---

## Prerequisites

* Windows 10/11 x64
* PowerShell 5 (or 7)
* NVIDIA GPU with CUDA 12.4+ (compute ≥ 7.0 highly recommended)
* \~40 GB free disk space (source tree and model)

---

## Scripts

### `install_ik_llamacpp.ps1`

Automates installation of build tools and compiles **`llama‑server.exe`**.

* Installs **Git**, **CMake**, **Ninja**, **Visual Studio 2022 Build Tools** and the latest **CUDA 12.4+** via `winget`.
* Clones [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) into **`vendor/`**.
* Builds the server executable with CUDA kernels for your GPU.

> **Tip ― choose the right CUDA arch**
> Pass the `‑CudaArch` switch to target your card exactly.
> E.g. Ampere (RTX 30‑series) ⇒ `‑CudaArch 86`.

| Architecture  | Cards (examples)   | Flag         |
| ------------- | ------------------ | ------------ |
| **Pascal**    | GTX 10×0, Quadro P | 60 / 61 / 62 |
| **Turing**    | RTX 20×0 / 16×0    | 75           |
| **Ampere**    | RTX 30×0           | 80 / 86 / 87 |
| **Ada**       | RTX 40×0           | 89           |
| **Blackwell** | RTX 50×0           | 90           |

Run from an **elevated** PowerShell prompt:

```powershell
Set-ExecutionPolicy Bypass -Scope Process
./install_ik_llamacpp.ps1 -CudaArch 86   # adjust as needed
```

### `run-qwen3-server.ps1`

* Downloads **`Qwen3‑Coder‑30B‑A3B‑Instruct‑1M‑IQ4_XS.gguf`** into `models/` (resumable BITS transfer).
* Launches **`llama‑server.exe`** with a tuned set of runtime flags.

```powershell
./run-qwen3-server.ps1
```

The server starts on [http://localhost:8080](http://localhost:8080) and exposes both a browser chat UI and an OpenAI‑compatible REST API.

---

## High‑Performance Inference with `ik_llama.cpp`

`ik_llama.cpp` adds several power‑user flags that do **not** exist in vanilla `llama.cpp`.  The script uses them out‑of‑the‑box – feel free to tweak.

| Flag                          | Purpose                                                                                                                                                                | Typical value(s)                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `‑ngl 99`                     | Offload **all** eligible layers to the first CUDA device ("99" is effectively “max”).                                                                                  | 99                                                   |
| `‑ctk <type>` / `‑ctv <type>` | **Quantise the runtime KV‑cache** (K or V half). Shrinks memory footprint & bandwidth.                                                                                 | `q8_0` (8‑bit), `f16` (default), `bf16`, …           |
| `‑ot '<regex>=<backend>'`     | **Override tensor placement** via a *regex* → backend map. Applied **after** `‑ngl`. Lets you keep huge MoE expert matrices on CPU while hot path tensors stay on GPU. | Examples:<br>`exps=CPU`<br>`blk\.[0-7]\.ffn.*=CUDA0` |
| `‑fa` / `‑‑flash‑attn`        | Enable Flash‑Attention kernels.                                                                                                                                        |                                                      |
| `‑fmoe` / `‑‑fused‑moe`       | Use fused MoE kernels for Qwen’s expert layers.                                                                                                                        |                                                      |
| `‑ser 1,0`                    | *Smart‑Expert‑Reduction*: compute only the most probable experts.                                                                                                      |                                                      |

### Example of the shipped launch line

```powershell
llama-server.exe `
  --model "models\Qwen3-Coder-30B-A3B-Instruct-1M-IQ4_XS.gguf" `
  -fa -fmoe -ser 1,0 `
  -c 65536 `
  -ctk q8_0 -ctv q8_0 `
  -ngl 99 `
  -ot "blk\.(0|1|..|19)\.ffn.*=CUDA0" `
  -ot exps=CPU `
  --threads 8 `
  --temp 0.6 --top-p 0.95 --top-k 20
```

* **`‑ctk/-ctv`** shrink the 64k‑context KV‑cache to one quarter of the f32 size while speeding up attention.
* **`‑ot`** pushes only the first 20 blocks’ MoE experts to GPU0; the remainder stay on RAM – a good balance for 24 GB cards.

Feel free to experiment: e.g. keep **V** in `f16` for slightly higher fidelity, or split blocks across multiple GPUs (`‑ot 'blk\.(0|1|2|3)=CUDA0' -ot 'blk\.(4|5|6|7)=CUDA1'`).

---

## Getting Started


1.  Open an **elevated** PowerShell window.
2.  Navigate to the directory containing the scripts.
3.  Run the installation script:
    ```powershell
    .\install_ik_llamacpp.ps1
    ```
4.  Once the installation is complete, run the server:
    ```powershell
    .\run-qwen3-server.ps1
    ```
5.  You can now interact with the model. The `llama-server` provides:
    - A web interface for interactive chat directly in your browser at `http://localhost:8080`.
    - An OpenAI-compatible API for programmatic access. You can send requests to endpoints like `http://localhost:8080/v1/chat/completions`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


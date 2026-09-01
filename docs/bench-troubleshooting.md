# Bench Troubleshooting

Historical forensics and reference material for bench/serving anomalies.
Operational how-to (prerequisites, experiment sequence, scripts) lives in
`docs/bench-runbook.md`; measured results in `docs/bench-results.md`.

---

## Tg variability — ROOT CAUSE FOUND AND FIXED

**Root cause:** `power-profiles-daemon` (KDE default) was setting a non-performance power profile on some boots, subtly degrading CPU/HWP state in a way that all sysfs checks (`governor`, `EPP`, `scaling_max_freq`, `cpu MHz`) still showed "performance" — yet tg ran at 32–35 t/s instead of ~39–40 t/s.

**Fix:** Replace `power-profiles-daemon` with `tuned-ppd` (CachyOS recommended):
```sh
sudo pacman -S tuned-ppd        # removes power-profiles-daemon automatically
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
```
After reboot: tg = **40.60 t/s** with zero preflight checks needed.

**Status update (2026-08-31):** tuned-ppd is no longer installed — the box is
back on stock PPD, currently holding `Profile=performance`. The fix's value
was proving the perf-profile mechanism was the root cause (the 40.60 t/s
result); it does not require tuned-ppd specifically. See the "CPU power
layering" section in `docs/bench-runbook.md` for the current-state caveat
(PPD owns the knobs again — re-verify sysfs after login/profile changes).

---

## Tg variability checklist (intermittent 33 t/s vs expected ~39 t/s) — archived

Symptom: tg varies between boots with no obvious cause. pp unaffected (GPU-bound).

| # | Check | How to verify | Status |
|---|-------|--------------|--------|
| 1 | CPU governor = performance | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | ✅ ruled out |
| 2 | EPP = performance | `cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference` | ✅ ruled out |
| 3 | CPU frequency boosting (4.5–4.9 GHz) | `grep "cpu MHz" /proc/cpuinfo \| sort -rn \| head -4` | ✅ ruled out |
| 4 | RAPL power limits (PL1=125W, PL2=241W) | `sudo cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_{0,1}_power_limit_uw` | ✅ ruled out |
| 5 | GPU VRAM pressure | `nvidia-smi \| grep MiB` — must be near-empty before bench | ✅ ruled out |
| 6 | Thermal throttling | `cat /sys/class/thermal/thermal_zone*/temp` — expect <70°C | ✅ ruled out |
| 7 | LTS kernel version | `uname -r` — same kernel both good and bad runs | ✅ ruled out |
| 8 | llama.cpp binary | build hash shown in bench output — same both sessions | ✅ ruled out |
| 9 | Background CPU load (Zed etc.) | `ps aux --sort=-%cpu \| head -8` | ✅ ruled out |
| 10 | cgroup CPU quota | `cat /sys/fs/cgroup/$(cat /proc/self/cgroup \| grep -o '[^:]*$' \| head -1)/cpu.max` — expect `max 100000` | ✅ ruled out |
| 11 | RAM speed | `sudo dmidecode -t memory \| grep "Configured Memory Speed"` — expect 5867 MT/s | ✅ ruled out |
| 12 | Deep C-states (C6/C7/C8/C10 wake latency) | `sudo cpupower idle-info` — disable with `sudo cpupower idle-set -D 1` then bench | ✅ ruled out — C10 (680µs), C8 (280µs), C6 (220µs) all disabled; tg unchanged |
| 13 | tuned-ppd throughput profile | `sudo pacman -S tuned && sudo tuned-adm profile throughput-performance` | ⬜ not yet tested |
| 14 | Intel HWP hardware response | May not follow SW governor correctly on some boots; `cat /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost` | ⬜ hwp_dynamic_boost=0 confirmed; HWP MSR state unverified (needs `sudo rdmsr -a 0x774`) |
| 15 | PCIe link speed | `nvidia-smi -q \| grep -A 3 "PCIe Generation"` — must be Gen 4 during bench | ✅ ruled out — Gen 4 confirmed live during tg phase |
| 16 | GPU SM clock drop during tg | `watch -n 0.2 "nvidia-smi -q -d CLOCK"` during bench tg phase | ✅ ruled out — 2520 MHz during tg (normal; GPU scales back due to low utilisation) |
| 17 | RAM speed on slow boot | `sudo dmidecode -t memory \| grep "Configured Memory Speed"` — run on a confirmed-slow boot | ✅ ruled out — 5867 MT/s confirmed on slow boot |
| 18 | Transparent Huge Pages | `cat /sys/kernel/mm/transparent_hugepage/enabled` | ✅ ruled out — [always] mode active |
| 19 | HWP MSR hardware EPP (bits 31–24 of 0x774) | `sudo modprobe msr && sudo rdmsr -a 0x774` — 0x00=performance, 0x80=balanced | ⬜ not yet verified — sysfs shows performance but HW may differ |
| 20 | power-profiles-daemon interfering | `systemctl status power-profiles-daemon` + `powerprofilesctl get` | ✅ **ROOT CAUSE** — replaced with tuned-ppd; tg jumped to 40.60 t/s on clean reboot |
| 21 | vm.swappiness=150 (ZRAM) pushing expert weights into ZRAM | `sysctl vm.swappiness` — CachyOS default is 150 with ZRAM | ⬜ confirmed 150; unlikely with 58 GiB free but worth checking on memory-pressured boots |
| 22 | energy_perf_bias per-core | `cat /sys/devices/system/cpu/cpu*/power/energy_perf_bias` — expect all 0 | ✅ ruled out — all P-cores at 0 (performance) |

---

## Common troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CUDA OOM at startup | ctx-size too large, parallel > 1, or too many layers on GPU | Reduce `--ctx-size`, set `--parallel 1`, or lower `-ngl` |
| RAM OOM / hard crash | Too many experts on CPU (e.g. N_CPU_MOE = total blocks) | Use fit or partial-cpu -ot; never send all experts to CPU on 64 GB |
| CUDA OOM mid-prompt (cuMemCreate) | GGML VMM pool can't grow — see "GGML CUDA memory pools" below | Increase `FIT_TARGET` to ≥1200 MiB, or set `GGML_CUDA_GRAPH_OPT=0` |
| pp collapses with ik_llama | CUDA graph compilation overhead | Pass `-fmoe 0` or switch back to llama.cpp |
| `_shexp` OOM with Qwen3.5-122B | Regex missing shared expert tensors | Use `(ch|)exps` not just `_exps` in all -ot patterns |
| Slow tg despite GPU offload | High active-param model (e.g. 122B 10B active) | This is architecture, not a tuning failure — switch to Qwen3-Coder-Next for speed |
| tg varies 27–40 t/s between boots | CPU power profile not set to performance | `echo performance \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` and `sudo powerprofilesctl set performance`. On CachyOS/KDE, `power-profiles-daemon` may reset EPP at login — make it persistent or run before bench. |

---

## GGML CUDA Memory Pools

Understanding this section prevents mysterious mid-session OOMs that don't appear at startup.

### How GGML allocates VRAM

GGML manages GPU memory in two pool implementations:

**VMM pool** (`ggml_cuda_pool_vmm`) — default on Turing+ hardware (RTX 20 series and newer):
- Uses CUDA Virtual Memory Management (`cuMemCreate` / `cuMemMap`).
- Reserves virtual address space in **1 GiB chunks**. Physical GPU memory is committed lazily on first access.
- When the pool needs to grow, it calls `cuMemCreate(&handle, 1_GiB, ...)`. If less than 1 GiB of free physical VRAM exists, this fails with `CUDA error: out of memory` — even if the actual allocation needed was only a few MB.
- Error site: `ggml-cuda.cu:489` function `alloc`.

**Legacy pool** (`ggml_cuda_pool_leg`) — fallback on older hardware:
- Uses plain `cudaMalloc`. Allocates only what's requested. No 1 GiB growth steps. Slower for many small allocs but never fails trying to reserve 1 GiB when you only needed 20 MB.

### Why fit's headroom margin matters for server use

`llama-bench` runs at a short context (512+128 = 640 tokens). At bench time, the VMM pool commits only the physical pages needed for that tiny context. All remaining VRAM stays uncommitted/free — `nvidia-smi` shows it as free.

`llama-fit-params` measures free VRAM at this same near-zero-load state and leaves `FIT_TARGET` MiB as margin. With `FIT_TARGET=128`, only 128 MiB of VRAM is intentionally reserved.

At server runtime with a growing conversation:
1. The KV cache fills with real tokens — new physical pages are committed lazily as the KV depth increases.
2. Scratch buffers for compute graphs may grow when a new context depth is hit for the first time (especially with `GGML_CUDA_GRAPH_OPT=1`, which recaptures CUDA graphs at new depths).
3. When any of these triggers a VMM pool growth attempt → `cuMemCreate(1 GiB)` → fails if < 1 GiB is physically free.

### Observed VRAM at runtime (RTX 4070, Q3CN, FIT_TARGET=512)

```
nvidia-smi at idle (server loaded, no active context):
  llama-server: 11182 MiB / 12282 MiB total (nvidia-smi)
  CUDA usable:  11875 MiB → free: 11875 - 11182 = 693 MiB physical
```

The 693 MiB free at idle is more than the 512 MiB FIT_TARGET because at idle the KV cache virtual space is reserved but physical pages haven't been committed yet. As context fills toward 64k, physical free shrinks toward ~500 MiB. The 693→500 MiB delta (~190 MiB) matches the KV cache for the first large prompt (~35k tokens × 2.2 KB/tok × 12 GQA layers ≈ ~140 MiB).

### Observed failure pattern

| Event | Tokens | Physical free | Status |
|-------|--------|--------------|--------|
| Server start | 0 | ~693 MiB | ✅ loads fine |
| First long prompt | ~35k | ~550 MiB | ✅ KV lazy commit, pool doesn't need to grow |
| Second longer prompt | ~44k | ~500 MiB | ❌ (old 128 MiB config) pool tries to grow → `cuMemCreate(1 GiB)` → OOM |
| Second longer prompt | ~44k | ~500 MiB | ✅ (current 512 MiB config + graph opt off) no growth needed |

The server process stays alive but the slot is in an error state when OOM occurs.

### Fixes

**Option 1 — Disable CUDA graph opt (recommended first try):**
```sh
# In llama-swap.yaml, set model "qwen3-coder-next" env:
#   - "GGML_CUDA_GRAPH_OPT=0"
systemctl --user restart llama-swap.service
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"say hi"}]}'
```
CUDA graph re-capture at new context depths is a significant VMM consumer. Disabling it removes this pressure at the cost of a small tg regression (typically < 1 t/s).

**Option 2 — Increase FIT_TARGET above 1 GiB:**
```sh
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 FIT_CTX=65536 FIT_TARGET=1536 \
  ./bench-models/bench-llama-cpp-qwen3-coder-next-fit.sh
```
Leaving ≥ 1200 MiB free ensures the VMM pool can always grow by one chunk. This typically costs 1–2 GPU layers (pp regression of ~14–30 t/s) but makes the server stable at any prompt depth up to ctx-size.

**Option 3 — Reduce batch size:**
```sh
--batch-size 1024 --ubatch-size 256
```
Smaller batches need smaller scratch buffers, reducing peak VMM pressure during prefill. May slightly reduce pp throughput.

### Rule of thumb

> If `FIT_TARGET < 1200 MiB`, set `GGML_CUDA_GRAPH_OPT=0`. The VMM pool grows in 1 GiB steps — any smaller headroom is a latent OOM waiting for a long enough prompt.

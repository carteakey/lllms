# Qwen3.8-Flash-Next — Internal Operations Doc

Internal record of every decision, measurement, and next step for serving
`Qwen3.8-Flash-Next` (qwen4exp) on `yeti-cachy`. Companion docs:
`AGENTS.md` (condensed rules), `CHANGELOG.md` (chronological record),
`docs/qwen3-8-flash-next-local-post.md` (published writeup),
`docs/bench-runbook.md §8` (bench numbers).

Status: 2026-08-31. Serving live via llama-swap, three tiers, all smoke-tested.

---

## 1. Hardware envelope (yeti-cachy)

| component | detail | implication |
| --- | --- | --- |
| GPU | RTX 4070, 12.28 GiB | KV + compute + fit-placed weights; full at 11.3/12.28 with fit-target 512 |
| RAM | 4×16 GB DDR5-4800 spec, running 5000 MT/s (i5-12600K, 2DPC) | random-access latency is the decode limiter; see §9.3 for clock plan |
| SSD | SN770 Gen4, ~6 GB/s | hosts the 38.4 GB n-gram shard; µs-latency vs ~50 ms/token budget → PLE offload is free |
| swap | zram 61.6 GiB (compressed, in-RAM) | zram metrics are noise for spill detection; use `read_bytes` |
| access | Tailscale 100.110.126.24, LAN 192.168.0.36, llama-swap :8080 | Bearer auth via `LLAMA_SWAP_API_KEY` |
| session | linger enabled (`Linger=yes`) | user manager survives graphical-session teardown → headless safe |

## 2. Quant decision (the full chain)

1. **Unsloth quants rejected on this box.** Unsloth bakes the 51.2 G-element
   PLE (n-gram) table into the weight shards: whole-file residency required.
   Header-parsed splits (remote range-fetch of shard headers):
   - UD-IQ4_XS: 92.3 GiB total = 29.8 GiB PLE (~4.65 bpw) + 62.5 GiB weights
   - UD-Q4_K_XL: 103.7 GiB total = 29.8 GiB PLE + ~74.7 GiB weights
   Fast-memory budget: 60 GB RAM + 12 GB VRAM ≈ 72 GB. IQ4_XS non-PLE (62.5)
   barely fits; Q4_K_XL (74.7) cannot → expert paging → the Reddit "5 t/s"
   failure mode. Zero headroom for long-conversation KV growth.
2. **Quality cross-check**: unsloth IQ4_XS (KLD 0.0836, top-1 89.55%) ≈
   AtomicChat AD-4.27bpw (0.0842, 89.49%) — a lateral move; only Q4_K_XL
   (0.0469, 92.26%) is a genuine step up, and it does not fit. "Unsloth
   Q4_XL or nothing" → nothing.
3. **AtomicChat AD-4.27bpw-Q4_K_M-M64 adopted** (33 shards, 92.9 GB file):
   PLE is its **own shard** — SSD offload is native via mmap (no
   `--override-tensor` needed, never `--no-mmap`/`--mlock`), 54.5 GB
   fast-memory footprint. Sidecar: their imatrix is public, recipe
   documented (asymmetric high-bit band blk 0-3 + 40-47, IQ4_NL ffn_down).
4. **The `-ot` technique is preserved knowledge**: for unsloth quants the
   equivalent offload is `-ot "per_layer_token_embd\.weight=CPU"` + default
   mmap. Documented for a bigger-RAM future, not used here.
5. **PLE depth does not matter much**: AtomicChat measured 6-bit vs 8.5-bit
   PLE → ΔKLD 0.0005. So BF16-PLE rebuilds (SassyDiffusion, 184.9 GB) are a
   lateral move at huge disk cost (§9.2).

## 3. Serving layout (three tiers)

| tier | entry | binary | commit | notes |
| --- | --- | --- | --- | --- |
| gold | `qwen38-flash-next` | `vendor/llama.cpp-master/build/bin/llama-server` | master, currently `e4b9af007` | tracks upstream; refresh = `git fetch` + rebuild (ccache ~2 min) |
| exp | `qwen38-flash-next-exp` | `vendor/llama.cpp-pr-test-28023-28068-27941/build/bin/llama-server-exp` | `e1748dbd5` (master + #28023 #28068 #27941) | A/B vs gold for unmerged fixes |
| MTP | `qwen38-flash-next-mtp` | same clone, `build/bin/llama-server` | `0b7d6d57d` (exp + #27836 + detached-head patch) | 32k ctx cap, opt-in |

- All three share the AtomicChat quant; `--parallel 1` mandatory everywhere
  (multi-slot corrupts the QSA indexer cache → hallucinations).
- **Binary-name duality gotcha**: the exp clone's `build/bin/llama-server` is
  the MTP binary, `llama-server-exp` is the exp binary. Branch switches in
  that clone require rebuilding both (see AGENTS.md).
- The detached-head MTP patch is a real commit now (was a dangling dirty
  file in the obsolete `pr-test-27836` clone).
- Collapse triggers: #28023/#28068/#27941 merged → delete exp delta;
  #27836 merged → MTP rides master; then only gold remains.

## 4. Production flags (gold) and why each exists

```bash
--fit on --fit-target 512     # auto placement; A/B beat -ncmoe 46 by +2.8% tg @64k
                              # (graphopt investigation). fit-target = MiB of
                              # VRAM left FREE — lower packs more onto GPU.
-c 65536 --parallel 1         # 64k ctx; multi-slot breaks qwen4exp indexer
-b 4096 -ub 1024              # measured optimum on this box
-fa on --jinja                # flash-attn + Qwen template (mandatory)
-ctk q8_0 -ctv q8_0           # KV 12/48 layers only (~24 KB/tok f16-class)
-t 10 --threads-batch 12      # physical cores; ~8+ saturates RAM random-access
--lazy-mode on                # renamed from --tensor-read-lazy (#27794 follow-ups);
                              # keeps PLE shard reads lazy (SSD)
--spec-type ngram-mod         # n-gram speculation; -match 60 -min 12 -max 24
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0   # Qwen thinking-mode recs
--reasoning-effort medium --reasoning-budget 4000 --reasoning-preserve
                              # xhigh default burns tokens; cap trades a small
                              # quality delta for large effective-tg gains
--no-warmup                   # first real request warms instead
```

MTP entry deltas: `-fit off -ngl 99 -ncmoe 46 -c 32768` (Q4_K_M draft head
occupies VRAM → ctx cap) + `--spec-type draft-mtp,ngram-mod
--spec-draft-n-max 2 --spec-draft-p-min 0.7 --spec-draft-ngl 99`.

## 5. Measurements ledger (2026-08-31, this box)

| metric | value | conditions |
| --- | --- | --- |
| tg (counting probe, spec warm) | 24.8-25.4 t/s balanced · **25.7-26.3 t/s with performance mode** | 64k, converged, spec-friendly text; +~3% from keeping uncore/P-cores hot between speculative verify rounds (measured 2026-08-31, post-`power-profiles-daemon` performance switch) |
| tg (novel prose, spec cold→warm) | 19.1 → 19.8 t/s | 192-tok story gens |
| tg (novel prose, converged) | **19.5 t/s** | 256-tok story, 1.3 MB reads = 5 KB/tok design floor |
| tg (prose, during spill) | 16.2-16.9 t/s | full box, expert re-faulting |
| pp (hot) | 198-200 t/s | 638-tok prompts |
| pp (cold first prompt) | −7.5 s expert fault-in | SN770 page-in; warmup gen cures |
| VRAM | 11.3/12.28 GiB | fit-target 512 achieved |
| RAM steady state | ~46 GB model resident + ~8 GB other; ~10-13 GiB slack | post page-in |
| spill signature | 8.6 GB/256tok (33.7 MB/tok) → converges to 1.3 MB | page-in of expert pool, NOT thrash |
| MTP (32k, code) | +15..26% tg, prose parity | p-min 0.7 gating, acceptance 0.81-0.97 |

Number identities: **16-17 t/s = spilling (unhealthy) · ~19.5 = healthy prose
floor · 24.8+ = speculation multiplier on predictable text** (pool 18.9 →
35.9 t/s on repeated identical prompts, +90%). Note the speculation pool
goes cold after unrelated generations — a single cold reading (e.g. 16.8)
is pool state, not a regression; re-probe 1-3× to re-warm.

CPU power profile: `performance` (governor + EPP). Effect measured: ~+3% on
the spec-warm path (ramp penalty between verify rounds), neutral on
sustained prose (loaded clocks were already pegged). Verify after reboot:
`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` → `performance`.

## 6. Memory model (the ledger)

92.9 GB file = 54.5 GB weights + 38.4 GB PLE shard.

```
VRAM  11.3 GiB : ~8 weights + ~1.5 KV(64k q8) + ~1.7 compute
RAM   ~46 GB   : mapped + lazy-cached weight pages (Cached 48.1, Mapped 47.0)
SSD   38.4 GB  : PLE shard, ~5 KB/token design traffic
other ~6-8 GB  : opencode, sshd, kernel
total working set ≈ 64 GB vs 61.6 RAM + 12.3 VRAM — tight by design
```

Decode per token: ~3.2 GB expert gathers from RAM (random access — the tg
limiter) + 5 KB from SSD. Prefill multiplies PLE reads by batch size →
220k-token prefill ≈ 18 min regardless (Gen4 SSD; Gen5 would buy here).

## 7. Ops runbook

- **Smoke test** (after any binary/flag change): SIGHUP → one chat request,
  `max_tokens >= 64` (6 tokens vanish into `<think>`; empty content ≠
  failure — check `reasoning_content`).
- **Spill check** (after any load, before long sessions): delta
  `/proc/<pid>/io read_bytes` across a 256-token gen. Healthy ≈ ≤1.5 MB
  total. Tens of MB/token = expert re-faulting. zram/free numbers are noise;
  `read_bytes` is ground truth. One-liner in AGENTS.md.
- **Page-in warm-up**: after ANY (re)load, expert pool pages in over ~5-10
  generations (8.6 GB → 0.5 GB per gen observed). Run 2-3 throwaway
  generations before judging speed.
- **Headless**: `sudo systemctl isolate multi-user.target` (recover:
  `graphical.target`); linger is on, llama-swap survives. Best clean state =
  fresh boot, no GUI apps.
- **Config changes**: snapshot yaml first (`llama-swap.yaml.bak-*`), SIGHUP,
  smoke, then consider it wired.
- **RAM hogs**: browser is the usual suspect (~40 GB zram-backed at times);
  kill before long sessions. RAM totals: model 46 + other 8 → do not stack
  desktop workloads on top of long generations without checking the spill
  one-liner.

## 8. Rejected / deferred (with numbers)

| option | reason |
| --- | --- |
| unsloth UD-Q4_K_XL | non-PLE 74.7 GiB > 72 GB fast memory; expert paging |
| unsloth UD-IQ4_XS | fits but quality-lateral to AD-4.27 (0.0836 vs 0.0842) |
| more layers on GPU | VRAM full (0.95 GiB free); needs KV shrink first |
| 128k ctx (q8 KV) | +1.55 GiB VRAM → evicts weights → spill returns |
| 128k via q5_1 KV | possible future arm (+~0.6 GiB); untested quality |
| vision (mmproj) | ~1-2 GiB VRAM + upstream multimodal broken (image positions not encoded) |
| SassyDiffusion PLEBF16 | 184.9 GB disk; PLE depth ≈ irrelevant (ΔKLD 0.0005); see §9.2 |
| AD-3.84bpw quant | frees ~9 GB RAM; KLD 0.2277 = the compromised tier |
| drop_caches / zram games | drops the model's own pages / net loss |

## 9. Next steps

### 9.1 MTP implementation (current tier: working, ahead of upstream)

**2026-08-31 rollback-stack experiment (concluded, reverted).** Context: PR
#28123 (recurrent-state rollback, CISC-approved) + the port PR
(#28118 on-device checkpoints, #28120 rollback enable, #28061 replay fix)
show 1.33-1.73x MTP multipliers on 80 GB-class cards (RTX PRO 6000: 83 →
144 t/s prose) — the host-path state serialization our old build partially
pays. Hand-merged all four onto our MTP branch (`pr-test-exp-mtp-rollback`,
78718f37e) and measured on this 12 GB box:

| config | code t/s | stability |
| --- | --- | --- |
| old build (host ckpt, ncmoe 46, Q4_K_M head) | **25.0** | stable |
| rollback, static 46 | — | CUDA OOM at first decode (graph capture) |
| rollback, static 48 | 32.8 once → 20.6 repro | OOM'd during later graph instantiate |
| rollback, fit 3072 | ~12 | stable but starved |
| rollback, fit 2048 | 14-16 | stable but starved |
| rollback, static 48 + Q3_K_M head (2.15 GB, requantized locally) | 20.9 → 20.6 | crashed on 2nd probe |

Findings: (a) the on-device checkpoint buffers need ~2.5-4 GB VRAM this
card does not have alongside the head + KV + compute; (b) the one 32.8
reading was ngram-pool resonance (the Fibonacci output was still in the
speculation pool), not steady state — treat one-off spec spikes as noise;
(c) the Q3_K_M requant (`agentionai-mtp-Q3_K_M.gguf`, kept on disk) did not
rescue VRAM and lowered acceptance; (d) hand-merging four PR heads with
divergent bases carries integration noise — net slower than the old build
at steady state. Reverted MTP tier to 0b7d6d57d (host checkpoints + p-min
0.7 gating = the right design for 12 GB). **Retry conditions**: #28123 +
#28118/#28120 merge into master (then gold refresh — no hand-merges), a
lighter draft head (unsloth layout via #28097 — note: unsloth has NOT
published a head; community heads dzannotti/ashbash/drluoto are gated/gone),
or bigger VRAM. The atomicchat org publishes no MTP head.

- [ ] Track #27836 + #28097 (draft-head-only GGUFs, unsloth layout) + the
      rollback stack above — when merged, rebuild master and collapse the
      MTP delta; then re-validate the 32k ctx cap (lighter head may allow
      48k/64k with MTP).
- [ ] Re-validate MTP on the exp stack after #28068-class fixes merge
      (draft head acceptance may shift with GDN l2norm change).
- [ ] Re-bench code vs prose acceptance with `--spec-draft-p-min` sweep
      (0.6/0.7/0.8) on the current stack — acceptance data is from the old
      build.

### 9.2 Full-precision n-gram (BF16 PLE)

Expected value is LOW: AtomicChat's own A/B (6-bit vs 8.5-bit PLE) moved
KLD by 0.0005 ≈ measurement error. BF16 should be strictly ≥ but likely
imperceptible. If pursued anyway:

- [ ] Disk math: SassyDiffusion PLEBF16-UD-Q4_K_XL = 184.9 GB vs ~120 GB
      free — needs cleanup or an external drive first.
- [ ] Graft option (experimental, precedent: `mtp-heads/graft-mtp-shard.py`):
      extract only the PLE tensors from the SassyDiffusion shards and
      graft onto the existing AD file, keeping our 92.9 GB layout. Verify
      tensor names/shapes match across publishers first
      (`maintenance/gguf_tensor_types.py`).
- [ ] Validation: perplexity A/B (wikitext, ctx 4096) AD-native vs grafted;
      only adopt if ΔPPL is outside ±1%. Expect it not to be.

### 9.3 RAM frequency increase (decode is RAM random-access bound) — DONE 2026-08-31

Current: 4×16 GB Corsair Vengeance DDR5-6000C36 (CMK32GX5M2E6000C36/
D6000C36), single-rank, ASRock B760M Steel Legend, i5-12600K, 2DPC.
Reboot 1 result: 5000 → **5600 MT/s** @ 1.35 V, user-set voltages above the
suggested ranges (stable across model soak). 

Measured impact (post-reboot, §5 protocol): prose 19.5 → **19.6-19.7**
(unique-prefix probes; flat), spec-warm 26.3 → 26.1 (flat), spill 0.2
MB/256tok (clean). +12% bandwidth bought ≈0-3% tg — confirms the workload
is random-access LATENCY bound, not bandwidth bound. The initial "20.1"
reading was a warm-cache sample; unique-prefix re-runs under performance
governor showed parity. Remaining RAM headroom: the kit's rated 6000 EXPO
profile (likely Gear 2 on Alder Lake 2DPC — latency tradeoff; given the
flat result at 5600, NOT worth pursuing for tg). Further tg gains must
come from algorithmic work (§9.4), not memory clocks.

Reboot 2 candidates (after RAM): re-bench §5 numbers; if tg moves >5%,
update runbook + AGENTS.md.

### 9.4 The 30 t/s campaign (target: 30 tok/s sustained)

Ladder measured 2026-08-31 (all warm, converged): gold prose 19.5 · gold
spec-warm 26.3 · MTP code 25.0 (32k cap). Gap analysis:

1. **MTP/ngram param sweep** (no reboot, ~30 min): `--spec-draft-p-min`
   0.5/0.6/0.7 × `--spec-draft-n-max` 2/3 × ngram-mod `-min 8 -max 32`.
   Best-known acceptance data is from the old build; the exp stack may
   accept differently. Also probe MTP's ceiling with the counting prompt.
2. **RAM clocks** (reboot 1, §9.3): +5-10% across every class →
   spec-warm ~28-29, code ~27-28.
3. **Upstream merges** (watch list §10): #27977 (tg-vs-ctx decay) +
   #27992 (O(log n) n-gram lookups) → refresh gold, re-bench at 64k;
   #27836 + #28097 → MTP on master, lighter head → MTP at 64k.
4. **True QSA sparsity** upstream: the last leg for prose 30.
5. Expected milestones: spec-warm 30 ≈ steps 1+2 (days); code 30 ≈ steps
   1-3 (weeks); prose 30 ≈ steps 2-4 (weeks, upstream-dependent).
   Prose 30 via local flags alone is NOT available — the 19.5 floor is
   RAM random-access bound (3.2 GB/token gathers); only bandwidth/latency
   (RAM clocks) or algorithmic cuts (upstream) move it.

### 9.5 Reboot checklist

1. Capture `sudo dmidecode -t 17` baseline (see §9.3.1) — or skip if RAM
   settings change is deferred.
2. Reboot into the chosen target (GUI or `multi-user.target`).
3. Verify llama-swap auto-alive (linger): `systemctl --user is-active llama-swap`.
3b. Verify CPU power profile survived: scaling_governor → `performance`
   (see §5); if it reset, either re-select Performance in the desktop
   applet or install the permanent fix once:
   `sudo cp maintenance/systemd/l3ms-cpufreq.service /etc/systemd/system/ &&
   sudo systemctl daemon-reload && sudo systemctl enable --now l3ms-cpufreq`
   (oneshot unit: sets governor+EPP=performance on every boot, survives
   desktop profile resets).
4. Load gold via router, run the page-in warm-up (2-3 throwaway gens), then
   the spill check — expect ≤1.5 MB/256tok.
5. Re-run the §5 quick numbers (counting probe ×2, story gen ×1) and diff
   against this doc.
6. Commit any deltas here.

## 10. Open watch items

- **#27977** (tg slowdown as ctx grows) and **#27992** (O(log n) n-gram
  lookups): target the long-ctx decode decay directly. When either merges,
  refresh gold + re-bench at 128k+ effective context.
- **#28068** (GDN l2norm max→rsqrt): under review — CISC skeptical,
  author's own numbers show marginal effect (KLD −1.7% rel, top-1 −0.2 pt).
  Stays in exp tier only; do not promote to gold unless it merges.
- **QSA true sparsity**: upstream still computes full attention then masks.
  When real sparsity lands, prefill is the step-change (18 min/220k →
  potentially minutes). Watch the PR list.
- **Multimodal**: broken upstream (positions not encoded). Keep qwen38
  entries text-only until a fix PR appears.
- **Unsloth quant rework**: community expects unsloth to re-ladder this
  model; if a future quant solves the PLE residency better than AD, redo
  the §2 math.

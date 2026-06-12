# /preflight

Run the system state preflight check before benchmarking or starting inference.
Based on `docs/bench-runbook.md` §2 — "System state check (run before every bench)".

## Usage

```
/preflight [--fix] [--bench]
```

**Arguments:**
- `--fix` — attempt to fix governor/EPP issues automatically (requires sudo)
- `--bench` — additional checks specific to benchmarking (background CPU load, VRAM usage)

## Checks

### 1. CPU governor (must be "performance")
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance
```
If not `performance`:
```bash
# Option A: via cpupower
sudo cpupower frequency-set -g performance

# Option B: via tuned-ppd (recommended for CachyOS — persistent across reboots)
sudo tuned-adm profile throughput-performance
```

### 2. EPP — Energy Performance Preference (must be "performance")
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
# Expected: performance
```
⚠️ `power-profiles-daemon` (KDE default) can silently set a non-performance HWP state even when sysfs shows `governor=performance`. This is the documented root cause of intermittent 32-35 t/s tg (expected ~40 t/s). If `power-profiles-daemon` is running, replace it:
```bash
sudo pacman -S tuned-ppd   # removes power-profiles-daemon automatically
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
# Then reboot for clean state
```

### 3. CPU frequency — P-cores must be near max boost
```bash
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -4
# Expected on i5-12600K: 4500–4900 MHz
```

### 4. RAM speed (critical for MoE models)
```bash
sudo dmidecode -t memory | grep -E "Speed|Configured"
# "Configured Memory Speed" must match your XMP/EXPO profile speed
```
⚠️ If `Configured Memory Speed` < `Speed`, XMP/EXPO is not enabled in BIOS.
For MoE models, RAM bandwidth IS token generation speed. DDR5 at 2000 MT/s = 3× slower tg than at 6000 MT/s.

### 5. VRAM state
```bash
nvidia-smi | grep MiB
# Expected: near-empty before bench (only driver reservation ~27 MiB)
```
If VRAM is occupied, unload any loaded models:
```bash
curl -X POST http://localhost:8080/models/unload -d '{"model":"<current-model>"}'
```

### 6. GPU clock during inference (tg phase)
```bash
nvidia-smi -q -d CLOCK | grep -A2 "Graphics"
# During tg: GPU SM clock scales back (e.g., 2520 MHz on RTX 4070)
# This is NORMAL — GPU is compute-idle during CPU expert processing
```

### 7. Background CPU load (bench only)
```bash
ps aux --sort=-%cpu | head -8
# Kill or pause Zed, browser, etc. before bench
```

### 8. Thermal state (bench only)
```bash
cat /sys/class/thermal/thermal_zone*/temp
# Expected: <70°C; throttling typically starts at 90–100°C
```

## Automated check script (run all at once)

```bash
echo "=== CPU Governor ===" && cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "=== EPP ===" && cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
echo "=== CPU MHz (top 4) ===" && grep "cpu MHz" /proc/cpuinfo | sort -rn | head -4
echo "=== RAM Speed ===" && sudo dmidecode -t memory | grep -E "Speed|Configured"
echo "=== VRAM ===" && nvidia-smi | grep MiB
echo "=== Thermals ===" && paste /sys/class/thermal/thermal_zone*/temp | awk '{for(i=1;i<=NF;i++) printf "zone%d: %d°C  ", i-1, $i/1000; print ""}'
echo "=== Top CPU procs ===" && ps aux --sort=-%cpu | head -6
```

## Expected healthy output

| Check | Expected value |
|-------|---------------|
| governor | `performance` |
| EPP | `performance` |
| CPU MHz (P-cores) | 4500–4900 MHz (i5-12600K) |
| Configured Memory Speed | 6000 MT/s (DDR5-6000 with XMP) |
| VRAM used | ~27 MiB (driver only) |
| Thermals | < 70°C |
| Background CPU | < 5% total non-system |

## Known intermittent issue

**Symptom:** tg reads 32–35 t/s instead of expected ~39–40 t/s. pp is unaffected (GPU-bound).

**Root cause:** `power-profiles-daemon` setting a non-performance HWP state on some boots.
All sysfs checks look clean but hardware is subtly degraded.

**Fix:** Replace `power-profiles-daemon` with `tuned-ppd` (see check #2 above) and reboot.
After fix: consistent 40.6 t/s on Qwen3-Coder-Next with zero preflight checks needed.

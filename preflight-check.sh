#!/usr/bin/env bash
# preflight-check.sh — verify system state before bench
# Usage: bash preflight-check.sh [--fix]
#   --fix   auto-apply safe remediations (needs sudo for some)
#
# Run on every boot before benchmarking to catch slow-tg root causes.
# Compare output across good vs bad boots to identify the culprit.

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; NC='\033[0m'; BOLD='\033[1m'
PASS="${GRN}PASS${NC}"; WARN="${YEL}WARN${NC}"; FAIL="${RED}FAIL${NC}"
FIX_MODE=0; [[ "$1" == "--fix" ]] && FIX_MODE=1

issues=0; fixes=()
warn()    { echo -e "  ${WARN}  $1"; ((issues++)); }
fail()    { echo -e "  ${FAIL}  $1"; ((issues++)); }
pass()    { echo -e "  ${PASS}  $1"; }
fix_hint(){ echo -e "         fix: $1"; fixes+=("$1"); }
section() { echo -e "\n${BOLD}── $1 ──${NC}"; }

apply_fix() {
  local cmd="$1"
  if [[ $FIX_MODE -eq 1 ]]; then
    echo -e "  ${YEL}→ applying:${NC} $cmd"
    eval "$cmd"
  fi
}

# ── Kernel ────────────────────────────────────────────────────────────────────
section "Kernel"
echo "       kernel : $(uname -r)"
echo "      cmdline : $(cat /proc/cmdline)"
MICROCODE=$(grep -m1 "microcode" /proc/cpuinfo | awk '{print $3}')
echo "    microcode : $MICROCODE"

# ── Power Profile Daemon ───────────────────────────────────────────────────────
section "Power Profile Daemon"
PPD_ACTIVE=$(systemctl is-active power-profiles-daemon 2>/dev/null)
TUNED_ACTIVE=$(systemctl is-active tuned 2>/dev/null)

if [[ "$TUNED_ACTIVE" == "active" ]]; then
  TUNED_PROFILE=$(tuned-adm active 2>/dev/null | awk -F': ' '{print $2}')
  echo "  backend : tuned (CachyOS recommended)"
  if [[ "$TUNED_PROFILE" == *"throughput"* || "$TUNED_PROFILE" == *"performance"* ]]; then
    pass "tuned profile = $TUNED_PROFILE"
  else
    warn "tuned profile = ${TUNED_PROFILE:-unknown} — want throughput-performance for bench"
    fix_hint "sudo tuned-adm profile throughput-performance"
    apply_fix "sudo tuned-adm profile throughput-performance"
  fi
elif [[ "$PPD_ACTIVE" == "active" ]]; then
  PPD_PROFILE=$(powerprofilesctl get 2>/dev/null)
  echo "  backend : power-profiles-daemon (CachyOS recommends switching to tuned-ppd)"
  if [[ "$PPD_PROFILE" == "performance" ]]; then
    pass "ppd profile = $PPD_PROFILE"
  else
    warn "ppd profile = ${PPD_PROFILE:-unknown} — want performance"
    fix_hint "sudo powerprofilesctl set performance"
    apply_fix "sudo powerprofilesctl set performance"
  fi
elif command -v powerprofilesctl &>/dev/null; then
  echo "  powerprofilesctl installed but no backend running (manual governor control — OK)"
  pass "no daemon active; governor/EPP controlled manually"
else
  pass "no power profile daemon active (manual governor control)"
fi

# ── CPU Governor & HWP ────────────────────────────────────────────────────────
section "CPU Governor & HWP"
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
EPP=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)
TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
SMT=$(cat /sys/devices/system/cpu/smt/active)
PSTATE=$(cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null)
HWP_BOOST=$(cat /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost 2>/dev/null)

[[ "$GOV" == "performance" ]] && pass "governor = $GOV" \
  || { fail "governor = $GOV (want: performance)"; fix_hint "sudo cpupower frequency-set -g performance"; apply_fix "sudo cpupower frequency-set -g performance"; }
[[ "$EPP" == "performance" ]] && pass "EPP = $EPP" \
  || { fail "EPP = $EPP (want: performance)"; fix_hint "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"; apply_fix "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null"; }
[[ "$TURBO" == "0" ]] && pass "turbo enabled" \
  || { fail "turbo DISABLED"; fix_hint "echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"; apply_fix "echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null"; }
[[ "$SMT" == "1" ]] && pass "SMT/HT active" \
  || fail "SMT disabled — only 6 logical CPUs (want 12 for THREADS=10)"
echo "  intel_pstate : $PSTATE  |  hwp_dynamic_boost: ${HWP_BOOST:-n/a}"

# energy_perf_bias — older ACPI hint, still active on Alder Lake alongside EPP
BIAS_VALS=$(cat /sys/devices/system/cpu/cpu{0..11}/power/energy_perf_bias 2>/dev/null | sort -u)
if [[ -n "$BIAS_VALS" ]]; then
  if echo "$BIAS_VALS" | grep -qvx "0"; then
    NON_ZERO=$(cat /sys/devices/system/cpu/cpu{0..11}/power/energy_perf_bias 2>/dev/null | grep -v "^0$" | wc -l)
    warn "energy_perf_bias: ${NON_ZERO} P-cores not at 0 (performance) — values: $(echo "$BIAS_VALS" | tr '\n' ' ')"
    fix_hint "echo 0 | sudo tee /sys/devices/system/cpu/cpu{0..11}/power/energy_perf_bias"
    apply_fix "for f in /sys/devices/system/cpu/cpu{0..11}/power/energy_perf_bias; do echo 0 | sudo tee \$f > /dev/null; done"
  else
    pass "energy_perf_bias = 0 (performance) on all P-cores"
  fi
else
  echo "  energy_perf_bias: not accessible"
fi

# ── CPU Frequency (P-cores 0–11) ─────────────────────────────────────────────
section "CPU Frequency (P-cores 0–11)"
FREQS=$(grep "cpu MHz" /proc/cpuinfo | awk '{print $4}' | head -12)
MIN_MHZ=$(echo "$FREQS" | sort -n  | head -1 | cut -d. -f1)
MAX_MHZ=$(echo "$FREQS" | sort -rn | head -1 | cut -d. -f1)
echo "  range at read: ${MIN_MHZ}–${MAX_MHZ} MHz  (idle expected; benchmark will boost to 4500 MHz all-core)"
SCALEMAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
CPUMAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
[[ "$SCALEMAX" == "$CPUMAX" ]] && pass "scaling_max uncapped = $(( CPUMAX / 1000 )) MHz" \
  || fail "scaling_max $(( SCALEMAX / 1000 )) MHz < cpuinfo_max $(( CPUMAX / 1000 )) MHz — artificially capped"

# ── C-States ─────────────────────────────────────────────────────────────────
section "C-States"
DEEP_ACTIVE=0
for state_dir in /sys/devices/system/cpu/cpu0/cpuidle/state*/; do
  name=$(cat "${state_dir}name" 2>/dev/null)
  disabled=$(cat "${state_dir}disable" 2>/dev/null)
  usage=$(cat "${state_dir}usage" 2>/dev/null)
  latency=$(cat "${state_dir}latency" 2>/dev/null)
  if [[ "$name" != "POLL" && "$disabled" == "0" && "$latency" -gt 50 ]] 2>/dev/null; then
    warn "Deep C-state ACTIVE: $name (latency ${latency}µs, entered ${usage}x this boot)"
    DEEP_ACTIVE=1
  else
    echo "        state : ${name} (latency ${latency}µs) — $([ "$disabled" == "1" ] && echo disabled || echo active)"
  fi
done
if [[ "$DEEP_ACTIVE" == "1" ]]; then
  fix_hint "sudo cpupower idle-set -D 1"
  apply_fix "sudo cpupower idle-set -D 1"
else
  pass "no deep C-states active (latency > 50µs)"
fi

# ── Memory ────────────────────────────────────────────────────────────────────
section "Memory"
TOTAL_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
FREE_GB=$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo)
SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null)
THP_ENABLED=$(cat /sys/kernel/mm/transparent_hugepage/enabled  | grep -o '\[.*\]' | tr -d '[]')
THP_DEFRAG=$(cat  /sys/kernel/mm/transparent_hugepage/defrag   | grep -o '\[.*\]' | tr -d '[]')
ANON_HUGE=$(awk '/AnonHugePages/{printf "%.0f MiB", $2/1024}' /proc/meminfo)

echo "    total RAM : ${TOTAL_GB} GiB  |  available: ${FREE_GB} GiB"
[[ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -gt 15000000 ]] \
  && pass "available RAM = ${FREE_GB} GiB (model needs ~15 GiB headroom)" \
  || { fail "available RAM = ${FREE_GB} GiB — too low, expert weights may swap"; fix_hint "kill memory-heavy processes before bench"; }

[[ "$THP_ENABLED" == "always" || "$THP_ENABLED" == "madvise" ]] \
  && pass "THP enabled = $THP_ENABLED" \
  || { warn "THP enabled = $THP_ENABLED (want: always or madvise)"; fix_hint "echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"; apply_fix "echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null"; }

# CachyOS tmpfiles sets defer+madvise (good for tcmalloc workloads)
[[ "$THP_DEFRAG" == "defer+madvise" || "$THP_DEFRAG" == "madvise" ]] \
  && pass "THP defrag  = $THP_DEFRAG" \
  || { warn "THP defrag  = $THP_DEFRAG (want: defer+madvise per CachyOS-Settings)"; fix_hint "echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"; apply_fix "echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null"; }

echo "  AnonHugePages : $ANON_HUGE"
if [[ -n "$SWAPPINESS" && "$SWAPPINESS" -le 10 ]]; then
  pass "vm.swappiness = $SWAPPINESS"
else
  warn "vm.swappiness = ${SWAPPINESS:-?} — CachyOS sets 150 for ZRAM but this aggressively compresses expert weight pages; decompression adds latency to every expert read during tg"
  fix_hint "sudo sysctl -w vm.swappiness=0   # only swap under OOM; safe for bench"
  apply_fix "sudo sysctl -w vm.swappiness=0"
fi

if command -v dmidecode &>/dev/null; then
  RAM_SPEED=$(sudo dmidecode -t memory 2>/dev/null | grep "Configured Memory Speed" | awk '{print $4, $5}' | sort -u)
  RAM_VOLT=$(sudo  dmidecode -t memory 2>/dev/null | grep "Configured Voltage"      | awk '{print $3, $4}' | sort -u)
  if [[ -n "$RAM_SPEED" ]]; then
    [[ "$RAM_SPEED" == *"5867"* ]] && pass "RAM speed = $RAM_SPEED (XMP applied)" \
      || fail "RAM speed = $RAM_SPEED — XMP not applied! Expected 5867 MT/s — fix in BIOS"
    echo "  RAM voltage : ${RAM_VOLT:-n/a}"
  else
    warn "dmidecode unavailable (run with sudo for RAM speed + voltage)"
  fi
fi

# ── RAPL Power Limits ─────────────────────────────────────────────────────────
section "RAPL Power Limits"
PL1_UW=$(cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_power_limit_uw 2>/dev/null)
PL2_UW=$(cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_1_power_limit_uw 2>/dev/null)
if [[ -n "$PL1_UW" ]]; then
  PL1_W=$(( PL1_UW / 1000000 )); PL2_W=$(( PL2_UW / 1000000 ))
  echo "  PL1 = ${PL1_W}W  |  PL2 = ${PL2_W}W"
  [[ "$PL1_W" -ge 100 ]] && pass "PL1 = ${PL1_W}W (expect ~125W)" \
    || fail "PL1 = ${PL1_W}W — too low, CPU will be silently throttled during expert compute"
  [[ "$PL2_W" -ge 150 ]] && pass "PL2 = ${PL2_W}W (expect ~241W)" \
    || fail "PL2 = ${PL2_W}W — too low"
else
  warn "RAPL not readable (needs sudo / powercap acl)"
fi

# ── GPU (NVIDIA) ──────────────────────────────────────────────────────────────
section "GPU (NVIDIA)"
if command -v nvidia-smi &>/dev/null; then
  GPU_PERSIST=$(nvidia-smi -q | grep "Persistence Mode" | awk '{print $4}')
  GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  GPU_PWR=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | cut -d. -f1)
  GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null)
  GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)

  echo "  persistence : $GPU_PERSIST  |  temp: ${GPU_TEMP}°C  |  power: ${GPU_PWR}W"
  echo "  VRAM        : ${GPU_MEM_USED} / ${GPU_MEM_TOTAL} MiB"

  [[ "$GPU_TEMP" -gt 85 ]] && fail "GPU temp ${GPU_TEMP}°C — thermal throttle risk" || pass "GPU temp ${GPU_TEMP}°C"
  [[ "$GPU_MEM_USED" -lt 500 ]] && pass "VRAM clear (${GPU_MEM_USED} MiB used)" \
    || warn "VRAM pre-occupied: ${GPU_MEM_USED} MiB — another process may interfere"

  # NVIDIA runtime power management (CachyOS udev sets "auto" which allows D3/Gen1 at idle)
  # nvidia-smi format: "GPU 00000000:01:00.0" → strip leading 00000000 → 0000:01:00.0
  GPU_PCI=$(nvidia-smi -q | grep "^GPU " | awk '{print tolower($2)}' | sed 's/^0*//' | sed 's/^/0000:/' 2>/dev/null | head -1)
  [[ -z "$GPU_PCI" ]] && GPU_PCI="0000:01:00.0"  # fallback for RTX 4070 on this system
  RUNTIME_PM="/sys/bus/pci/devices/${GPU_PCI}/power/control"
  if [[ -f "$RUNTIME_PM" ]]; then
    PM_CTRL=$(cat "$RUNTIME_PM")
    if [[ "$PM_CTRL" == "auto" ]]; then
      warn "NVIDIA runtime PM = auto (CachyOS udev default) — GPU may enter D3/PCIe Gen1 at idle; PCIe renegotiates on bench start but adds latency"
      fix_hint "echo on | sudo tee $RUNTIME_PM   # disables runtime PM for this boot"
    else
      pass "NVIDIA runtime PM = on (no D3 downclocking)"
    fi
  fi

  # NVreg_DynamicPowerManagement: 0x02 (fine-grained, mobile) in CachyOS modprobe is wrong for desktop
  NV_DPM=$(cat /sys/module/nvidia/parameters/DynamicPowerManagement 2>/dev/null)
  if [[ -n "$NV_DPM" ]]; then
    if [[ "$NV_DPM" == "0" ]]; then
      pass "NVreg_DynamicPowerManagement = 0 (disabled, correct for desktop)"
    elif [[ "$NV_DPM" == "2" ]]; then
      warn "NVreg_DynamicPowerManagement = 2 (fine-grained/mobile PM) — CachyOS modprobe.d default; suboptimal for desktop RTX 4070"
      fix_hint "echo 'options nvidia NVreg_DynamicPowerManagement=0' | sudo tee /etc/modprobe.d/nvidia-desktop.conf  # then reboot"
    else
      echo "  NVreg_DynamicPowerManagement = $NV_DPM"
    fi
  fi

  echo ""
  echo "  PCIe gen must be checked under load:"
  echo "    watch -n 0.2 \"nvidia-smi -q | grep -A 3 'PCIe Generation'\""
else
  warn "nvidia-smi not found"
fi

# ── Storage I/O Scheduler ─────────────────────────────────────────────────────
section "Storage I/O Scheduler"
MODEL_DEFAULT="/mnt/lab/models"
MODEL_PATH="${MODEL:-$MODEL_DEFAULT}"
# Resolve the block device backing the model path
if [[ -d "$MODEL_PATH" ]] || [[ -f "$MODEL_PATH" ]]; then
  MOUNT_DEV=$(df "$MODEL_PATH" 2>/dev/null | tail -1 | awk '{print $1}')
  # Strip partition number to get base device
  BASE_DEV=$(echo "$MOUNT_DEV" | sed 's|/dev/||; s|[0-9]*$||; s|p[0-9]*$||')
  SCHED_FILE="/sys/block/${BASE_DEV}/queue/scheduler"
  if [[ -f "$SCHED_FILE" ]]; then
    SCHED=$(cat "$SCHED_FILE" | grep -o '\[.*\]' | tr -d '[]')
    IS_ROTATIONAL=$(cat "/sys/block/${BASE_DEV}/queue/rotational" 2>/dev/null)
    echo "  model path  : $MODEL_PATH → /dev/${BASE_DEV} (rotational=${IS_ROTATIONAL:-?})"
    if [[ "$IS_ROTATIONAL" == "0" ]]; then
      # NVMe: none, SATA SSD: mq-deadline (per CachyOS udev rules)
      if [[ "$BASE_DEV" == nvme* ]]; then
        [[ "$SCHED" == "none" ]] && pass "I/O scheduler = $SCHED (correct for NVMe)" \
          || warn "I/O scheduler = $SCHED (want: none for NVMe per CachyOS udev rules)"
      else
        [[ "$SCHED" == "mq-deadline" ]] && pass "I/O scheduler = $SCHED (correct for SATA SSD)" \
          || warn "I/O scheduler = $SCHED (want: mq-deadline for SATA SSD per CachyOS udev rules)"
      fi
    else
      [[ "$SCHED" == "bfq" ]] && pass "I/O scheduler = $SCHED (correct for HDD)" \
        || warn "I/O scheduler = $SCHED (want: bfq for HDD per CachyOS udev rules)"
    fi
  else
    echo "  model path  : $MODEL_PATH (scheduler path not found for ${BASE_DEV})"
  fi
else
  echo "  model path  : $MODEL_PATH not mounted/found — skip I/O scheduler check"
fi

# ── Background CPU Load ───────────────────────────────────────────────────────
section "Background CPU Load"
# Exclude known harmless: kernel threads, claude (this script's session), terminal emulators
HIGH_LOAD=$(ps aux --sort=-%cpu | awk 'NR>1 && $3>1.0 {print $0}' \
  | grep -v -E '\[.*\]|claude|kgx|konsole|alacritty|kitty|foot|bash|zsh|fish|sshd' \
  | awk '{printf "    %-30s %s%%\n", $11, $3}' | head -8)
if [[ -n "$HIGH_LOAD" ]]; then
  warn "processes using >1% CPU (excluding terminal/shell/claude):"
  echo "$HIGH_LOAD"
else
  pass "no unexpected processes using >1% CPU"
fi

# ── BIOS ──────────────────────────────────────────────────────────────────────
section "BIOS"
if command -v dmidecode &>/dev/null; then
  BIOS_INFO=$(sudo dmidecode -t bios 2>/dev/null | grep -E "Vendor|Version|Release Date" | sed 's/^\s*/  /')
  [[ -n "$BIOS_INFO" ]] && echo -e "$BIOS_INFO" || warn "BIOS info unavailable (needs sudo)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════${NC}"
if [[ "$issues" -eq 0 ]]; then
  echo -e "${GRN}${BOLD}  All checks passed — good to bench.${NC}"
else
  echo -e "${YEL}${BOLD}  $issues issue(s) found.${NC}"
  if [[ $FIX_MODE -eq 0 && ${#fixes[@]} -gt 0 ]]; then
    echo ""
    echo -e "  Run ${BOLD}bash preflight-check.sh --fix${NC} to auto-apply safe remediations."
    echo "  Manual fixes still needed:"
    for f in "${fixes[@]}"; do
      [[ "$f" == *"BIOS"* || "$f" == *"reboot"* || "$f" == *"modprobe"* ]] && echo "    • $f"
    done
  fi
fi
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo ""

#!/usr/bin/env bash
# L3MS: force performance CPU policy at boot (survives desktop profile resets).
# Checks intel_pstate active: 'powersave' governor + EPP=performance is the
# full-boost combination there; we set both explicitly.
set -euo pipefail

for pol in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -d "$pol" ] || continue
    echo performance > "$pol/scaling_governor" 2>/dev/null || true
    [ -f "$pol/energy_performance_preference" ] && \
        echo performance > "$pol/energy_performance_preference" 2>/dev/null || true
done
echo "L3MS: cpufreq governor/EPP set to performance"

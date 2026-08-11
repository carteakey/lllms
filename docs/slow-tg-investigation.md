# Slow TG boot investigation

CAR-145 remains externally blocked until the KPC host provides one fast and one
slow boot captured with the privileged preflight. The repository does not run
that command from a developer workstation because it requires host hardware and
`sudo` access.

On KPC, capture both boots with the same model, llama.cpp revision, and power
state. Every fast and slow sample must include its timestamp, hostname, boot
ID, and kernel so that the two runs can be tied to the exact host boot:

```bash
capture_preflight() {
  label="$1"
  {
    printf 'timestamp: '; date --iso-8601=seconds
    printf 'hostname: '; hostname
    printf 'boot_id: '; cat /proc/sys/kernel/random/boot_id
    printf 'kernel: '; uname -r
    sudo bash preflight-check.sh
  } | tee "/tmp/l3ms-preflight-${label}.txt"
}

capture_preflight fast
# reproduce a slow TG boot, then run the same command:
capture_preflight slow
diff -u /tmp/l3ms-preflight-fast.txt /tmp/l3ms-preflight-slow.txt
```

The privileged preflight covers the CPU governor, EPP, current frequency,
PCIe link and runtime power state, GPU state, RAM availability and speed,
transparent huge pages (THP), and C-state activity. Keep those checks in both
samples; do not compare only the benchmark throughput.

Record the diff in the private Forge project context or the CAR-145 Linear
issue; do not commit host identifiers, credentials, or raw production output.
The existing Rust telemetry covers process CPU/RAM/GPU, disk-free, and network
bytes, but it cannot determine governor, EPP, PCIe, clock, RAM, THP, or C-state
causes without those two host captures.

# Slow TG boot investigation

CAR-145 remains externally blocked until the KPC host provides one fast and one
slow boot captured with the privileged preflight. The repository does not run
that command from a developer workstation because it requires host hardware and
`sudo` access.

On KPC, capture both boots with the same model, llama.cpp revision, and power
state:

```bash
sudo bash preflight-check.sh | tee /tmp/l3ms-preflight-fast.txt
# reproduce a slow TG boot, then run the same command:
sudo bash preflight-check.sh | tee /tmp/l3ms-preflight-slow.txt
diff -u /tmp/l3ms-preflight-fast.txt /tmp/l3ms-preflight-slow.txt
```

Record the diff in the private Forge project context or the CAR-145 Linear
issue; do not commit host identifiers, credentials, or raw production output.
The existing Rust telemetry covers process CPU/RAM/GPU, disk-free, and network
bytes, but it cannot determine governor, EPP, PCIe, clock, RAM, THP, or C-state
causes without those two host captures.

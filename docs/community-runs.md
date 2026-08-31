# Community runs

Community benchmark results live in `docs/community-runs.json` and must satisfy `docs/community-runs.schema.json`. They are displayed separately from the local RTX 4070 profiles and never participate in that ranking.

Every submission needs a public HTTPS source containing the raw output or enough detail to review the result. Historical local profiles may show missing evidence because those fields were not recorded at the time; new community runs cannot omit them.

Example shape:

```json
{
  "id": "example-24gb-run",
  "submittedAt": "2026-07-17",
  "sourceUrl": "https://example.com/raw-benchmark",
  "hardware": {
    "gpu": "Example GPU",
    "gpuCount": 1,
    "vramGb": 24,
    "cpu": "Example CPU",
    "ramGb": 64,
    "ramSpeed": "DDR5-6000",
    "interconnect": null
  },
  "software": {
    "os": "Linux",
    "backend": "CUDA",
    "llamaCppCommit": "0123456789abcdef",
    "driver": "000.00"
  },
  "model": {
    "name": "org/model",
    "quant": "Q4_K_M",
    "modelUrl": "https://huggingface.co/org/model"
  },
  "benchmark": {
    "command": "llama-bench -m model.gguf -p 512 -n 128 -r 5",
    "testedContext": 65536,
    "cacheState": "cold",
    "promptTokens": 512,
    "generatedTokens": 128,
    "repetitions": 5
  },
  "metrics": {
    "pp": 1000.0,
    "tg": 50.0,
    "draftAcceptance": null
  },
  "notes": "Optional constraints or stability observations."
}
```

The generator performs dependency-free structural checks. A JSON Schema validator can apply the stricter types, ranges, and formats before review.

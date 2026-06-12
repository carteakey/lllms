import subprocess
import time
import re
import os

# Paths to models
MODELS_BASE = "/home/kchauhan/models/unsloth"
LLAMA_CLI = "./vendor/llama.cpp/build/bin/llama-cli"

CONFIGS = [
    {
        "name": "Gemma 4 26B Baseline (Q5_K_XL)",
        "model": f"{MODELS_BASE}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
        "draft": None,
    },
    {
        "name": "Gemma 4 26B QAT (Q4_K_XL)",
        "model": f"{MODELS_BASE}/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "draft": None,
    },
    {
        "name": "Gemma 4 26B MTP Only (Q5_K_XL + Assistant)",
        "model": f"{MODELS_BASE}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
        "draft": f"{MODELS_BASE}/gemma-4-26B-A4B-it-qat-GGUF/mtp-gemma-4-26B-A4B-it.gguf",
        "draft_n_max": "2"
    },
    {
        "name": "Gemma 4 26B QAT + MTP (Q4_K_XL + Assistant)",
        "model": f"{MODELS_BASE}/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "draft": f"{MODELS_BASE}/gemma-4-26B-A4B-it-qat-GGUF/mtp-gemma-4-26B-A4B-it.gguf",
        "draft_n_max": "2"
    },
    {
        "name": "Gemma 4 12B Ultrafast (Q4_K_XL + MTP)",
        "model": f"{MODELS_BASE}/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "draft": f"{MODELS_BASE}/gemma-4-12B-it-qat-GGUF/mtp-gemma-4-12B-it.gguf",
        "draft_n_max": "4"
    }
]

PROMPT = "Write a comprehensive guide on the history of artificial intelligence, focusing on the recent breakthroughs in large language models and multi-token prediction."

def run_bench(config):
    cmd = [
        LLAMA_CLI,
        "-m", config["model"],
        "-p", PROMPT,
        "-n", "256",
        "-t", "10",
        "-fa", "on",
        "--fit", "on",
        "-fitt", "1536",
        "-st",
        "--simple-io"
    ]
    
    if config.get("draft"):
        cmd += [
            "--model-draft", config["draft"],
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", config.get("draft_n_max", "2")
        ]
        
    print(f"\n>>> Running: {config['name']}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stderr + result.stdout
        
        # Extract TPS
        tps_match = re.search(r"Generation.*?([0-9.]+)\s*t/s", output)
        if tps_match:
            tps = float(tps_match.group(1))
            print(f"Result: {tps} tok/s")
            return tps
        else:
            print("Failed to find TPS in output")
            if "out of memory" in output.lower():
                print("Error: Out of Memory")
            print(f"Output preview: {output[-500:]}")
            return None
            
    except Exception as e:
        print(f"Error running bench: {e}")
        return None

def main():
    results = []
    for config in CONFIGS:
        tps = run_bench(config)
        results.append((config["name"], tps))
        
    print("\n\n" + "="*50)
    print("FINAL PERFORMANCE RESULTS")
    print("="*50)
    print(f"{'Configuration':<50} | {'TPS':<10}")
    print("-" * 63)
    for name, tps in results:
        tps_str = f"{tps:.2f}" if tps else "N/A"
        print(f"{name:<50} | {tps_str:<10}")

if __name__ == "__main__":
    main()

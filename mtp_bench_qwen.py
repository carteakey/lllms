#!/usr/bin/env python3
import subprocess
import time
import re
import sys

# Paths to models
MODELS_BASE = "/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
LLAMA_CLI = "./vendor/llama.cpp/build/bin/llama-cli"

Q4_MODEL = f"{MODELS_BASE}/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
Q6_MODEL = f"{MODELS_BASE}/Qwen3.6-35B-A3B-UD-Q6_K.gguf"

# Define the runs we want to evaluate
CONFIGS = [
    # --- Q4 Variant ---
    {
        "model_path": Q4_MODEL,
        "model_name": "Q4",
        "name": "Qwen 3.6 35B Q4 Baseline (No MTP)",
        "use_mtp": False,
        "draft_n_max": None,
        "enable_thinking": True
    },
    {
        "model_path": Q4_MODEL,
        "model_name": "Q4",
        "name": "Qwen 3.6 35B Q4 MTP (n-max = 1)",
        "use_mtp": True,
        "draft_n_max": 1,
        "enable_thinking": True
    },
    {
        "model_path": Q4_MODEL,
        "model_name": "Q4",
        "name": "Qwen 3.6 35B Q4 MTP (n-max = 2)",
        "use_mtp": True,
        "draft_n_max": 2,
        "enable_thinking": True
    },
    {
        "model_path": Q4_MODEL,
        "model_name": "Q4",
        "name": "Qwen 3.6 35B Q4 MTP (n-max = 3)",
        "use_mtp": True,
        "draft_n_max": 3,
        "enable_thinking": True
    },
    {
        "model_path": Q4_MODEL,
        "model_name": "Q4",
        "name": "Qwen 3.6 35B Q4 MTP (n-max = 2, NoThink)",
        "use_mtp": True,
        "draft_n_max": 2,
        "enable_thinking": False
    },
    
    # --- Q6 Variant ---
    {
        "model_path": Q6_MODEL,
        "model_name": "Q6",
        "name": "Qwen 3.6 35B Q6 Baseline (No MTP)",
        "use_mtp": False,
        "draft_n_max": None,
        "enable_thinking": True
    },
    {
        "model_path": Q6_MODEL,
        "model_name": "Q6",
        "name": "Qwen 3.6 35B Q6 MTP (n-max = 1)",
        "use_mtp": True,
        "draft_n_max": 1,
        "enable_thinking": True
    },
    {
        "model_path": Q6_MODEL,
        "model_name": "Q6",
        "name": "Qwen 3.6 35B Q6 MTP (n-max = 2)",
        "use_mtp": True,
        "draft_n_max": 2,
        "enable_thinking": True
    },
    {
        "model_path": Q6_MODEL,
        "model_name": "Q6",
        "name": "Qwen 3.6 35B Q6 MTP (n-max = 1, NoThink)",
        "use_mtp": True,
        "draft_n_max": 1,
        "enable_thinking": False
    }
]

PROMPT = "Explain how speculative decoding works in large language model inference, in three short paragraphs."

def run_bench(config):
    # Base command using taskset to run on cores 0-11 and matching llama-swap parameters
    cmd = [
        "taskset", "-c", "0-11",
        LLAMA_CLI,
        "-m", config["model_path"],
        "-p", PROMPT,
        "-n", "256",
        "-t", "10",
        "-fa", "on",
        "--fit", "on",
        "--fit-ctx", "131072",
        "--fit-target", "512",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--no-mmap",
        "--mlock",
        "--temp", "0.6",
        "--top-p", "0.95",
        "--top-k", "20",
        "--min-p", "0.00",
        "-st",
        "--simple-io"
    ]
    
    if config["use_mtp"]:
        cmd += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(config["draft_n_max"])
        ]
        
    if not config["enable_thinking"]:
        cmd += [
            "--chat-template-kwargs", '{"enable_thinking":false}'
        ]
        
    print(f"\n>>> Running: {config['name']}")
    try:
        # We run the command and capture stderr + stdout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stderr + result.stdout
        
        # Extract TPS (Tokens Per Second)
        tps_match = re.search(r"Generation.*?([0-9.]+)\s*t/s", output)
        prompt_match = re.search(r"Prompt.*?([0-9.]+)\s*t/s", output)
        
        tps = float(tps_match.group(1)) if tps_match else None
        prompt_tps = float(prompt_match.group(1)) if prompt_match else None
        
        if tps:
            print(f"Result: {tps} tok/s (Prompt: {prompt_tps if prompt_tps else 'N/A'} t/s)")
            return {"tps": tps, "prompt_tps": prompt_tps, "error": None}
        else:
            print("Failed to find Generation TPS in output")
            if "out of memory" in output.lower():
                print("Error: Out of Memory")
                return {"tps": None, "prompt_tps": None, "error": "OOM"}
            
            # Try to print some debug info
            print("--- Output Tail ---")
            lines = output.splitlines()
            for line in lines[-20:]:
                print(line)
            return {"tps": None, "prompt_tps": None, "error": "Failed to parse"}
            
    except subprocess.TimeoutExpired:
        print("Error: Benchmark timed out")
        return {"tps": None, "prompt_tps": None, "error": "Timeout"}
    except Exception as e:
        print(f"Error running bench: {e}")
        return {"tps": None, "prompt_tps": None, "error": str(e)}

def main():
    results = []
    for config in CONFIGS:
        res = run_bench(config)
        results.append((config["name"], res))
        time.sleep(2)
        
    print("\n\n" + "="*85)
    print("COMPREHENSIVE PERFORMANCE RESULTS FOR QWEN 3.6 35B MTP (Q4 vs Q6, Thinking vs NoThink)")
    print("="*85)
    print(f"{'Configuration':<42} | {'Gen (t/s)':<12} | {'Prompt (t/s)':<14} | {'Status':<10}")
    print("-" * 90)
    for name, res in results:
        tps_str = f"{res['tps']:.2f}" if res['tps'] is not None else "N/A"
        prompt_str = f"{res['prompt_tps']:.2f}" if res['prompt_tps'] is not None else "N/A"
        status_str = "SUCCESS" if res['error'] is None else res['error']
        print(f"{name:<42} | {tps_str:<12} | {prompt_str:<14} | {status_str:<10}")

if __name__ == "__main__":
    main()

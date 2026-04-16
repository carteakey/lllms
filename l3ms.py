#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from l3ms.script_store import command_for_script

ROOT = Path(__file__).resolve().parent
BENCH_SCRIPT_GLOB = "bench-models/bench-llama-cpp-*.sh"


def collect_bench_scripts() -> List[Path]:
    return sorted([p for p in ROOT.glob(BENCH_SCRIPT_GLOB) if p.is_file()])


def pretty_name(path: Path) -> str:
    name = path.stem
    for prefix in ("bench-ik-llama-cpp-", "bench-llama-cpp-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def print_bench_list(scripts: List[Path]) -> None:
    if not scripts:
        print("No bench scripts found")
        return
    print(f"BENCH scripts ({len(scripts)}):")
    for idx, script in enumerate(scripts, start=1):
        rel = script.relative_to(ROOT).as_posix()
        print(f"  {idx:>2}. {pretty_name(script):<28} {rel}")


def _import_llama_swap():
    try:
        from l3ms import llama_swap
        return llama_swap
    except ModuleNotFoundError as exc:
        if exc.name == "httpx":
            print("Error: httpx is not installed.")
            print("Install it with: python3 -m pip install -r requirements-tui.txt")
        else:
            print(f"Error importing llama_swap: {exc}")
        return None


async def fetch_swap_models(llama_swap):
    return await llama_swap.list_models()


def print_run_list() -> int:
    llama_swap = _import_llama_swap()
    if llama_swap is None:
        return 1
    try:
        models = asyncio.run(fetch_swap_models(llama_swap))
    except Exception as exc:
        print(f"llama-swap unreachable at {llama_swap.DEFAULT_BASE_URL}: {exc}")
        print("Start llama-swap.service (see docs/llama-swap-runbook.md).")
        return 1
    if not models:
        print("No models reported by llama-swap")
        return 0
    print(f"RUN models ({len(models)}):")
    for idx, model in enumerate(models, start=1):
        label = f"{model.id:<42} state={model.state}"
        if model.name:
            label += f"  {model.name}"
        print(f"  {idx:>2}. {label}")
    return 0


def choose_bench(scripts: List[Path]) -> Optional[Path]:
    if not scripts:
        return None
    print_bench_list(scripts)
    print("Select script index to run, or 'q' to quit.")
    while True:
        raw = input("> ").strip()
        if not raw and len(scripts) == 1:
            return scripts[0]
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(scripts):
                return scripts[idx]
        print(f"Invalid selection: {raw!r}. Enter 1-{len(scripts)} or q.")


def choose_swap_model(models) -> Optional[str]:
    if not models:
        return None
    print(f"RUN models ({len(models)}):")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx:>2}. {model.id:<42} state={model.state}")
    print("Select model index to load, or 'q' to quit.")
    while True:
        raw = input("> ").strip()
        if not raw and len(models) == 1:
            return models[0].id
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx].id
        print(f"Invalid selection: {raw!r}. Enter 1-{len(models)} or q.")


def stream_command(cmd: List[str]) -> int:
    print(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping process...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return proc.wait()


def interactive_bench(filter_text: str, extra_raw: str) -> int:
    scripts = collect_bench_scripts()
    if filter_text:
        lowered = filter_text.lower()
        scripts = [s for s in scripts if lowered in s.relative_to(ROOT).as_posix().lower()]
    if not scripts:
        print(f"No bench scripts found for filter: {filter_text!r}")
        return 1
    script = choose_bench(scripts)
    if script is None:
        print("Cancelled.")
        return 0
    try:
        extra_args = shlex.split(extra_raw) if extra_raw.strip() else []
    except ValueError as exc:
        print(f"Invalid --extra value: {exc}")
        return 2
    code = stream_command(command_for_script(script, extra_args))
    print(f"Exited with code {code}")
    return code


def interactive_run(filter_text: str) -> int:
    llama_swap = _import_llama_swap()
    if llama_swap is None:
        return 1
    try:
        models = asyncio.run(fetch_swap_models(llama_swap))
    except Exception as exc:
        print(f"llama-swap unreachable at {llama_swap.DEFAULT_BASE_URL}: {exc}")
        return 1
    if filter_text:
        lowered = filter_text.lower()
        models = [m for m in models if lowered in m.id.lower() or lowered in m.name.lower()]
    if not models:
        print(f"No models match filter: {filter_text!r}")
        return 1
    model_id = choose_swap_model(models)
    if model_id is None:
        print("Cancelled.")
        return 0

    print(f"POST {llama_swap.DEFAULT_BASE_URL}/models/load  model={model_id}")
    try:
        result = asyncio.run(llama_swap.load_model(model_id))
    except Exception as exc:
        print(f"load failed: {exc}")
        return 1
    print(result)
    print(f"\nChat: curl {llama_swap.DEFAULT_BASE_URL}/v1/chat/completions -H 'Content-Type: application/json' \\")
    print(f"           -d '{{\"model\":\"{model_id}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'")
    return 0


def launch_tui() -> int:
    try:
        from l3ms import L3MSApp
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            print("Error: textual is not installed.")
            print("Install it with: python3 -m pip install -r requirements-tui.txt")
            return 1
        raise
    app = L3MSApp()
    app.run()
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L3MS launcher (TUI + interactive run/bench CLI)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run",
        nargs="?",
        const="",
        metavar="FILTER",
        help="Pick a model from llama-swap and trigger POST /models/load",
    )
    group.add_argument(
        "--bench",
        nargs="?",
        const="",
        metavar="FILTER",
        help="Interactive bench-script picker (optional substring filter)",
    )
    group.add_argument(
        "--list",
        choices=["run", "bench", "all"],
        help="List available models (run) / bench scripts and exit",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra args appended to bench script command for --bench",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.list:
        rc = 0
        if args.list in {"run", "all"}:
            rc = print_run_list() or rc
        if args.list in {"bench", "all"}:
            if args.list == "all":
                print()
            print_bench_list(collect_bench_scripts())
        raise SystemExit(rc)

    if args.run is not None:
        raise SystemExit(interactive_run(args.run))

    if args.bench is not None:
        raise SystemExit(interactive_bench(args.bench, args.extra))

    raise SystemExit(launch_tui())


if __name__ == "__main__":
    main()

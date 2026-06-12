#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from l3ms.script_store import command_for_script

ROOT = Path(__file__).resolve().parent
RUN_SCRIPT_GLOB = "run-models/run-llama-cpp-*.sh"
BENCH_SCRIPT_GLOB = "bench-models/bench-llama-cpp-*.sh"


def print_quickstart() -> None:
    print("L3MS quick start")
    print()
    print("  1) Open the TUI")
    print("     python3 l3ms.py")
    print("     - Start tab opens first with guided actions")
    print()
    print("  2) Run without entering TUI")
    print("     python3 l3ms.py --run")
    print("     python3 l3ms.py --bench")
    print()
    print("  3) Discover available scripts")
    print("     python3 l3ms.py --list all")
    print()
    print("  4) Pass extra args to selected script")
    print('     python3 l3ms.py --run qwen --extra "--ctx-size 32768"')


def collect_scripts(mode: str) -> List[Path]:
    pattern = RUN_SCRIPT_GLOB if mode == "run" else BENCH_SCRIPT_GLOB
    return sorted([path for path in ROOT.glob(pattern) if path.is_file()])


def pretty_name(path: Path) -> str:
    name = path.stem
    for prefix in ("run-llama-cpp-", "bench-llama-cpp-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def print_script_list(mode: str, scripts: List[Path]) -> None:
    if not scripts:
        print(f"No {mode} scripts found")
        return
    print(f"{mode.upper()} scripts ({len(scripts)}):")
    for idx, script in enumerate(scripts, start=1):
        rel = script.relative_to(ROOT).as_posix()
        print(f"  {idx:>2}. {pretty_name(script):<28} {rel}")


def choose_script(mode: str, scripts: List[Path]) -> Optional[Path]:
    if not scripts:
        return None

    print_script_list(mode, scripts)
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


def interactive_run(mode: str, filter_text: str, extra_raw: str) -> int:
    scripts = collect_scripts(mode)
    if filter_text:
        lowered = filter_text.lower()
        scripts = [s for s in scripts if lowered in s.relative_to(ROOT).as_posix().lower()]

    if not scripts:
        print(f"No {mode} scripts found for filter: {filter_text!r}")
        return 1

    script = choose_script(mode, scripts)
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
        help="Interactive run-mode script picker (optional substring filter)",
    )
    group.add_argument(
        "--bench",
        nargs="?",
        const="",
        metavar="FILTER",
        help="Interactive bench-mode script picker (optional substring filter)",
    )
    group.add_argument(
        "--list",
        choices=["run", "bench", "all"],
        help="List available scripts and exit",
    )
    group.add_argument(
        "--quickstart",
        action="store_true",
        help="Print a quick-start guide and exit",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra args appended to script command for --run/--bench",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.quickstart:
        print_quickstart()
        raise SystemExit(0)

    if args.list:
        if args.list in {"run", "all"}:
            print_script_list("run", collect_scripts("run"))
        if args.list in {"bench", "all"}:
            if args.list == "all":
                print()
            print_script_list("bench", collect_scripts("bench"))
        raise SystemExit(0)

    if args.run is not None:
        raise SystemExit(interactive_run("run", args.run, args.extra))

    if args.bench is not None:
        raise SystemExit(interactive_run("bench", args.bench, args.extra))

    raise SystemExit(launch_tui())


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shlex
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .config_store import (
    DEFAULT_CONFIG_PATH,
    csv_to_list,
    list_versions,
    load_config,
    normalize_model,
    restore_version,
    save_config,
    validate_config,
)
from .script_store import (
    list_script_versions,
    load_script,
    restore_script_version,
    save_script_with_version,
)

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = ROOT / "model_downloader" / "download_hf_model.py"
RUN_SCRIPT_GLOB = "run-models/run-llama-cpp-*.sh"
IK_RUN_SCRIPT_GLOB = "run-models/run-ik-llama-cpp-*.sh"
BENCH_SCRIPT_GLOB = "bench-models/bench-llama-cpp-*.sh"
MAINTENANCE_SCRIPT_GLOB = "maintenance/*.sh"


def collect_scripts(pattern: str) -> List[Path]:
    return sorted([path for path in ROOT.glob(pattern) if path.is_file()])


def command_for_script(path: Path, extra_args: List[str]) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".sh":
        return ["bash", str(path), *extra_args]
    if suffix == ".ps1":
        return ["pwsh", "-File", str(path), *extra_args]
    if suffix in {".bat", ".cmd"}:
        return ["cmd", "/c", str(path), *extra_args]
    return ["bash", str(path), *extra_args]


def collect_llama_binaries() -> List[tuple]:
    """Find all llama-server binaries under vendor/*/build/bin/."""
    options: List[tuple] = [("auto (script default)", "")]
    for binary in sorted(ROOT.glob("vendor/*/build/bin/llama-server")):
        if binary.is_file() and os.access(str(binary), os.X_OK):
            vendor_name = binary.relative_to(ROOT).parts[1]
            options.append((vendor_name, str(binary)))
    return options


class DownloadPanel(Static):
    def __init__(self) -> None:
        super().__init__(id="download_panel")
        self.config_path = DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {"base_models_dir": "", "models": []}
        self.selected_index: Optional[int] = None
        self.active_download: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Input(value=str(self.config_path), id="config_path")
                yield Button("Load", id="btn_load")
                yield Button("Save", id="btn_save", variant="success")
                yield Button("Validate", id="btn_validate")
                yield Select([], id="version_select", prompt="Versions")
                yield Button("Restore", id="btn_restore")

            with Horizontal(classes="row"):
                yield Input(placeholder="base_models_dir", id="base_models_dir")
                yield Checkbox("slow preset (4 workers)", value=True, id="slow_flag")
                yield Input(placeholder="max workers override", id="max_workers")
                yield Input(placeholder="save note", id="save_note")
                yield Static("💾 —", id="disk_space_label")

            with Horizontal(classes="row main"):
                with Vertical(classes="left"):
                    yield Label("Models")
                    yield DataTable(id="models_table")
                    with Horizontal(classes="row"):
                        yield Button("Add", id="btn_add")
                        yield Button("Apply Edit", id="btn_apply")
                        yield Button("Delete", id="btn_delete", variant="error")
                    with Horizontal(classes="row"):
                        yield Button("Download Selected", id="btn_download_selected")
                        yield Button("Download Enabled", id="btn_download_enabled")
                    yield Label(
                        "Keys: Alt+T table, Alt+I editor, Alt+O load, Alt+W save, Alt+V validate, "
                        "Alt+N add, Alt+A apply, Alt+K delete, Alt+D selected, Alt+E enabled, Alt+Y clear log"
                    )

                with Vertical(classes="right"):
                    yield Label("Model Editor")
                    yield Checkbox("enabled", id="m_enabled", value=True)
                    yield Input(placeholder="repo_id", id="m_repo_id")
                    yield Input(placeholder="description", id="m_description")
                    yield Input(placeholder="local_dir", id="m_local_dir")
                    yield Input(placeholder="revision", id="m_revision")
                    yield Input(placeholder="allow_patterns (comma separated)", id="m_allow")
                    yield Input(placeholder="ignore_patterns (comma separated)", id="m_ignore")
                    yield Checkbox("force_download", id="m_force", value=False)
                    yield Input(placeholder="max_workers (blank = null)", id="m_workers")

            yield Label("Activity Log")
            yield RichLog(id="activity_log", wrap=True, markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#models_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "enabled", "repo_id", "pattern", "local_dir")
        self.load_current_config()
        self.focus_table()
        self.run_worker(self.refresh_disk_space(), exclusive=False)

    async def refresh_disk_space(self) -> None:
        """Update the disk space label for the target drive."""
        label = self.query_one("#disk_space_label", Static)
        target = self.query_one("#base_models_dir", Input).value.strip()
        # Walk up to find first existing ancestor (drive may be partially mounted)
        path = Path(target) if target else None
        if path:
            check = path
            while check and not check.exists():
                check = check.parent if check.parent != check else None
        else:
            check = None
        if not check:
            label.update("💾 —")
            return
        try:
            usage = shutil.disk_usage(check)
            free_gb = usage.free / 1_073_741_824
            total_gb = usage.total / 1_073_741_824
            label.update(f"💾 {free_gb:.0f} / {total_gb:.0f} GB free  [{check}]")
        except OSError:
            label.update(f"⚠ drive not mounted  [{target}]")

    async def _estimate_download_size(self, repo_id: str, allow_patterns: List[str],
                                      ignore_patterns: List[str]) -> Optional[int]:
        """Return total bytes of remote files matching patterns, or None on error."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            total = 0
            for f in api.list_repo_tree(repo_id, recursive=True):
                name = getattr(f, "path", "") or getattr(f, "rfilename", "")
                size = getattr(f, "size", None)
                if size is None:
                    continue
                # Apply allow_patterns filter
                if allow_patterns and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                    continue
                # Apply ignore_patterns filter
                if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
                    continue
                total += size
            return total if total > 0 else None
        except Exception:
            return None


    def set_status(self, message: str) -> None:
        self.query_one("#activity_log", RichLog).write(message)

    def focus_table(self) -> None:
        self.query_one("#models_table", DataTable).focus()

    def focus_editor(self) -> None:
        self.query_one("#m_repo_id", Input).focus()

    def clear_log(self) -> None:
        self.query_one("#activity_log", RichLog).clear()
        self.set_status("Download log cleared")

    def parse_speed_args(self) -> List[str]:
        args: List[str] = []
        max_workers = self.query_one("#max_workers", Input).value.strip()
        if max_workers:
            if not max_workers.isdigit() or int(max_workers) <= 0:
                raise ValueError("max workers override must be a positive integer")
            args.extend(["--max-workers", max_workers])
            return args
        if self.query_one("#slow_flag", Checkbox).value:
            args.append("--slow")
        return args

    def _update_version_select(self) -> None:
        select = self.query_one("#version_select", Select)
        versions = list_versions(self.config_path)
        select.set_options([(name, name) for name in versions])

    def _set_selected_index(self, idx: int) -> None:
        if not (0 <= idx < len(self.config.get("models", []))):
            return
        self.selected_index = idx

    def _sync_selection_from_cursor(self) -> None:
        table = self.query_one("#models_table", DataTable)
        if self.config.get("models") and 0 <= table.cursor_row < len(self.config["models"]):
            self._set_selected_index(table.cursor_row)

    def load_current_config(self) -> None:
        path_value = self.query_one("#config_path", Input).value.strip()
        if path_value:
            self.config_path = Path(path_value).expanduser()

        self.config = load_config(self.config_path)
        self.query_one("#base_models_dir", Input).value = self.config.get("base_models_dir", "")
        self.selected_index = 0 if self.config.get("models") else None
        self.refresh_models_table()
        self.load_model_into_editor(self.selected_index)
        self._update_version_select()
        self.set_status(f"Loaded config: {self.config_path}")
        self.run_worker(self.refresh_disk_space(), exclusive=False)

    def refresh_models_table(self) -> None:
        table = self.query_one("#models_table", DataTable)
        table.clear()
        models = self.config.get("models", [])
        for i, model in enumerate(models):
            allow = model.get("allow_patterns") or []
            pattern = allow[0] if allow else "*"
            table.add_row(
                str(i),
                "yes" if model.get("enabled", True) else "no",
                model.get("repo_id", ""),
                pattern,
                model.get("local_dir", ""),
                key=str(i),
            )

        if models:
            if self.selected_index is None or self.selected_index >= len(models):
                self.selected_index = 0
            table.move_cursor(row=self.selected_index, column=0)
        else:
            self.selected_index = None

    def model_from_editor(self) -> Dict[str, Any]:
        workers_raw = self.query_one("#m_workers", Input).value.strip()
        workers = None
        if workers_raw:
            if not workers_raw.isdigit() or int(workers_raw) <= 0:
                raise ValueError("model max_workers must be a positive integer or blank")
            workers = int(workers_raw)

        return normalize_model(
            {
                "enabled": self.query_one("#m_enabled", Checkbox).value,
                "repo_id": self.query_one("#m_repo_id", Input).value,
                "description": self.query_one("#m_description", Input).value,
                "local_dir": self.query_one("#m_local_dir", Input).value,
                "revision": self.query_one("#m_revision", Input).value,
                "allow_patterns": csv_to_list(self.query_one("#m_allow", Input).value),
                "ignore_patterns": csv_to_list(self.query_one("#m_ignore", Input).value),
                "force_download": self.query_one("#m_force", Checkbox).value,
                "max_workers": workers,
            }
        )

    def load_model_into_editor(self, idx: Optional[int]) -> None:
        model = {}
        if idx is not None and 0 <= idx < len(self.config.get("models", [])):
            model = self.config["models"][idx]

        self.query_one("#m_enabled", Checkbox).value = bool(model.get("enabled", True))
        self.query_one("#m_repo_id", Input).value = str(model.get("repo_id", ""))
        self.query_one("#m_description", Input).value = str(model.get("description", ""))
        self.query_one("#m_local_dir", Input).value = str(model.get("local_dir", ""))
        self.query_one("#m_revision", Input).value = str(model.get("revision", ""))
        self.query_one("#m_allow", Input).value = ", ".join(model.get("allow_patterns") or [])
        self.query_one("#m_ignore", Input).value = ", ".join(model.get("ignore_patterns") or [])
        self.query_one("#m_force", Checkbox).value = bool(model.get("force_download", False))
        self.query_one("#m_workers", Input).value = str(model.get("max_workers") or "")

    def apply_editor_to_selected(self) -> None:
        if self.selected_index is None:
            raise ValueError("no model selected")
        self.config["models"][self.selected_index] = self.model_from_editor()
        self.refresh_models_table()

    def validate_current_config(self) -> bool:
        try:
            self._sync_selection_from_cursor()
            if self.selected_index is not None:
                self.apply_editor_to_selected()
        except ValueError as exc:
            self.set_status(f"Validation error: {exc}")
            return False

        self.config["base_models_dir"] = self.query_one("#base_models_dir", Input).value.strip()
        errors = validate_config(self.config)
        if errors:
            self.set_status("Validation failed:")
            for err in errors:
                self.set_status(f"- {err}")
            return False
        self.set_status("Config validation passed")
        return True

    def save_current_config(self) -> bool:
        try:
            self._sync_selection_from_cursor()
            if self.selected_index is not None:
                self.apply_editor_to_selected()
            self.config["base_models_dir"] = self.query_one("#base_models_dir", Input).value.strip()
            note = self.query_one("#save_note", Input).value.strip() or "manual-save"
            save_config(self.config_path, self.config, note=note)
            self._update_version_select()
            self.set_status(f"Saved config: {self.config_path}")
            return True
        except ValueError as exc:
            self.set_status(f"Save failed: {exc}")
            return False

    def restore_selected_version(self) -> None:
        selected = self.query_one("#version_select", Select).value
        if not isinstance(selected, str) or not selected:
            self.set_status("No version selected")
            return
        try:
            restore_version(self.config_path, selected)
            self.load_current_config()
            self.set_status(f"Restored version: {selected}")
        except ValueError as exc:
            self.set_status(f"Restore failed: {exc}")

    def add_model(self) -> None:
        self.config.setdefault("models", []).append(
            {
                "enabled": True,
                "repo_id": "",
                "local_dir": "",
                "allow_patterns": [],
                "ignore_patterns": [],
                "revision": "",
                "force_download": False,
                "max_workers": None,
                "description": "",
            }
        )
        self.selected_index = len(self.config["models"]) - 1
        self.refresh_models_table()
        self.load_model_into_editor(self.selected_index)
        self.set_status("Added model row")

    def apply_model_edit(self) -> None:
        try:
            self._sync_selection_from_cursor()
            self.apply_editor_to_selected()
            self.set_status("Applied editor changes to selected model")
        except ValueError as exc:
            self.set_status(f"Apply failed: {exc}")

    def delete_selected_model(self) -> None:
        self._sync_selection_from_cursor()
        if self.selected_index is None:
            self.set_status("No model selected")
            return
        if not (0 <= self.selected_index < len(self.config.get("models", []))):
            self.set_status("Invalid selection")
            return

        self.config["models"].pop(self.selected_index)
        if not self.config["models"]:
            self.selected_index = None
        elif self.selected_index >= len(self.config["models"]):
            self.selected_index = len(self.config["models"]) - 1

        self.refresh_models_table()
        self.load_model_into_editor(self.selected_index)
        self.set_status("Deleted model")

    async def run_download_command(self, cmd: List[str]) -> None:
        self.set_status(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self.set_status(line.decode("utf-8", errors="replace").rstrip())
        code = await proc.wait()
        self.set_status(f"Download exited with code {code}")
        await self.refresh_disk_space()

    async def download_selected_model(self) -> None:
        if self.active_download and not self.active_download.done():
            self.set_status("A download is already running")
            return

        try:
            self._sync_selection_from_cursor()
            if self.selected_index is not None:
                self.apply_editor_to_selected()
            else:
                self.set_status("No model selected")
                return

            model = self.config["models"][self.selected_index]
            repo_id = model.get("repo_id", "")
            if not repo_id:
                self.set_status("Selected model has no repo_id")
                return

            allow = model.get("allow_patterns") or []
            ignore = model.get("ignore_patterns") or []

            # Show size estimate before starting
            self.set_status(f"Checking remote file sizes for {repo_id}…")
            est = await self._estimate_download_size(repo_id, allow, ignore)
            if est is not None:
                est_gb = est / 1_073_741_824
                usage_val = self.query_one("#disk_space_label", Static).renderable
                self.set_status(f"  ~{est_gb:.1f} GB to pull  ·  {usage_val}")
            else:
                self.set_status("  (could not estimate remote size)")

            cmd = ["python3", str(DOWNLOAD_SCRIPT), "--repo-id", str(repo_id)]
            local_dir = str(model.get("local_dir", "")).strip()
            if local_dir:
                cmd.extend(["--local-dir", local_dir])

            if allow:
                cmd.append("--allow-patterns")
                cmd.extend([str(x) for x in allow])

            if ignore:
                cmd.append("--ignore-patterns")
                cmd.extend([str(x) for x in ignore])

            revision = str(model.get("revision", "")).strip()
            if revision:
                cmd.extend(["--revision", revision])

            if model.get("force_download", False):
                cmd.append("--force-download")

            cmd.extend(self.parse_speed_args())
            self.active_download = asyncio.create_task(self.run_download_command(cmd))
            await self.active_download
        except ValueError as exc:
            self.set_status(f"Download command failed: {exc}")

    async def download_enabled_models(self) -> None:
        if self.active_download and not self.active_download.done():
            self.set_status("A download is already running")
            return

        cmd = ["python3", str(DOWNLOAD_SCRIPT), "--config", str(self.config_path)]
        try:
            cmd.extend(self.parse_speed_args())
            self.active_download = asyncio.create_task(self.run_download_command(cmd))
            await self.active_download
        except ValueError as exc:
            self.set_status(f"Download command failed: {exc}")

    @on(DataTable.RowHighlighted, "#models_table")
    def on_model_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self._set_selected_index(idx)
        self.load_model_into_editor(idx)

    @on(DataTable.RowSelected, "#models_table")
    def on_model_row(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self._set_selected_index(idx)
        self.load_model_into_editor(idx)

    @on(Button.Pressed, "#btn_load")
    def on_load(self) -> None:
        self.load_current_config()

    @on(Button.Pressed, "#btn_validate")
    def on_validate(self) -> None:
        self.validate_current_config()

    @on(Button.Pressed, "#btn_save")
    def on_save(self) -> None:
        self.save_current_config()

    @on(Button.Pressed, "#btn_restore")
    def on_restore(self) -> None:
        self.restore_selected_version()

    @on(Button.Pressed, "#btn_add")
    def on_add(self) -> None:
        self.add_model()

    @on(Button.Pressed, "#btn_apply")
    def on_apply(self) -> None:
        self.apply_model_edit()

    @on(Button.Pressed, "#btn_delete")
    def on_delete(self) -> None:
        self.delete_selected_model()

    @on(Button.Pressed, "#btn_download_selected")
    async def on_download_selected(self) -> None:
        await self.download_selected_model()

    @on(Button.Pressed, "#btn_download_enabled")
    async def on_download_enabled(self) -> None:
        await self.download_enabled_models()


class RunPanel(Static):
    class JobStarted(Message):
        def __init__(self, name: str, started: str, mode: str) -> None:
            super().__init__()
            self.name = name
            self.started = started
            self.mode = mode

    class JobFinished(Message):
        def __init__(self, name: str, elapsed: str, exit_code: int, mode: str) -> None:
            super().__init__()
            self.name = name
            self.elapsed = elapsed
            self.exit_code = exit_code
            self.mode = mode

    def __init__(self) -> None:
        super().__init__(id="run_panel")
        self.mode = "run"
        self.run_scripts: List[Path] = []
        self.bench_scripts: List[Path] = []
        self.filtered: List[Path] = []
        self.selected_script: Optional[Path] = None
        self.running_proc: Optional[asyncio.subprocess.Process] = None
        self.running_task: Optional[asyncio.Task[None]] = None
        self.resource_task: Optional[asyncio.Task[None]] = None
        self.running_started_at: Optional[float] = None
        self._current_job_name: str = "idle"

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Label("Mode")
                yield Select([("Run", "run"), ("Bench", "bench")], value="run", id="run_mode")
                yield Button("Refresh", id="run_refresh")
                yield Button("Start (Ctrl+R)", id="run_start", variant="success")
                yield Button("Stop (Ctrl+S)", id="run_stop", variant="error")
            with Horizontal(classes="row"):
                yield Label("Binary")
                yield Select([], id="run_binary_select", prompt="auto (script default)")
                yield Button("Scan", id="run_binary_scan")
            with Horizontal(classes="row"):
                yield Static("Current: idle", id="run_current_model")
                yield Static("Resources: idle", id="run_resources")

            with Horizontal(classes="row"):
                yield Input(placeholder="filter scripts (Ctrl+F)", id="run_filter")
                yield Input(placeholder="extra args appended to script", id="run_extra_args")

            with Horizontal(classes="row main"):
                with Vertical(classes="left"):
                    yield DataTable(id="run_scripts_table")
                    yield Label(
                        "Keys: Ctrl+F filter, Ctrl+J table, Ctrl+U editor, Ctrl+M toggle mode, "
                        "Ctrl+R start, Ctrl+S stop, Ctrl+P save script, Ctrl+L clear log"
                    )
                    yield RichLog(id="run_log", wrap=True, markup=False)

                with Vertical(classes="right"):
                    yield Label("Script Editor")
                    yield Static("No script selected", id="run_selected_path")
                    with Horizontal(classes="row"):
                        yield Select([], id="run_version_select", prompt="Script versions")
                        yield Input(placeholder="save note", id="run_save_note")
                    with Horizontal(classes="row"):
                        yield Button("Reload", id="run_edit_reload")
                        yield Button("Save", id="run_edit_save", variant="success")
                        yield Button("Restore", id="run_edit_restore")
                    yield TextArea("", id="run_editor")

    def on_mount(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "script")
        self.refresh_script_inventory()
        self.refresh_binary_selector()
        self.focus_table()

    def set_status(self, message: str) -> None:
        self.query_one("#run_log", RichLog).write(message)

    def focus_filter(self) -> None:
        self.query_one("#run_filter", Input).focus()

    def focus_table(self) -> None:
        self.query_one("#run_scripts_table", DataTable).focus()

    def focus_editor(self) -> None:
        self.query_one("#run_editor", TextArea).focus()

    def clear_log(self) -> None:
        self.query_one("#run_log", RichLog).clear()
        self.set_status("Run log cleared")

    def set_runtime_state(self, model: str, resources: str) -> None:
        self.query_one("#run_current_model", Static).update(model)
        self.query_one("#run_resources", Static).update(resources)

    def refresh_binary_selector(self) -> None:
        binaries = collect_llama_binaries()
        select = self.query_one("#run_binary_select", Select)
        select.set_options(binaries)
        self.set_status(f"Found {len(binaries) - 1} llama-server binary/binaries in vendor/")

    def refresh_script_inventory(self) -> None:
        self.run_scripts = sorted(
            collect_scripts(RUN_SCRIPT_GLOB) + collect_scripts(IK_RUN_SCRIPT_GLOB)
        )
        self.bench_scripts = collect_scripts(BENCH_SCRIPT_GLOB)
        self.refresh_table()

    def current_scripts(self) -> List[Path]:
        return self.bench_scripts if self.mode == "bench" else self.run_scripts

    def refresh_table(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        filter_text = self.query_one("#run_filter", Input).value.strip().lower()
        scripts = self.current_scripts()
        self.filtered = []

        table.clear()
        for script in scripts:
            rel = script.relative_to(ROOT).as_posix()
            if filter_text and filter_text not in rel.lower():
                continue
            self.filtered.append(script)
            idx = len(self.filtered) - 1
            table.add_row(str(idx), rel, key=str(idx))

        if self.filtered:
            self.selected_script = self.filtered[0]
            table.move_cursor(row=0, column=0)
            self.load_selected_script_into_editor()
            self.set_status(
                f"Loaded {len(self.filtered)} {self.mode} script(s) "
                f"({len(scripts)} total before filter)"
            )
        else:
            self.selected_script = None
            self.query_one("#run_selected_path", Static).update("No script selected")
            self.query_one("#run_editor", TextArea).text = ""
            self.query_one("#run_version_select", Select).set_options([])
            self.set_status(f"No {self.mode} scripts match current filter")

    def toggle_mode(self) -> None:
        select = self.query_one("#run_mode", Select)
        self.mode = "bench" if self.mode == "run" else "run"
        select.value = self.mode
        self.refresh_table()

    def selected_model_name(self) -> str:
        if self.selected_script is None:
            return "idle"
        name = self.selected_script.stem
        for prefix in ("run-ik-llama-cpp-", "bench-ik-llama-cpp-", "run-llama-cpp-", "bench-llama-cpp-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    def _sync_selected_from_cursor(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        if self.filtered and 0 <= table.cursor_row < len(self.filtered):
            self.selected_script = self.filtered[table.cursor_row]

    def load_selected_script_into_editor(self) -> None:
        if self.selected_script is None:
            return
        try:
            content = load_script(self.selected_script)
        except ValueError as exc:
            self.set_status(f"Failed to load script: {exc}")
            return

        rel = self.selected_script.relative_to(ROOT).as_posix()
        self.query_one("#run_selected_path", Static).update(rel)
        self.query_one("#run_editor", TextArea).text = content

        versions = list_script_versions(self.selected_script)
        select = self.query_one("#run_version_select", Select)
        select.set_options([(name, name) for name in versions])

    def save_editor_script(self) -> None:
        self._sync_selected_from_cursor()
        if self.selected_script is None:
            self.set_status("No script selected")
            return

        note = self.query_one("#run_save_note", Input).value.strip() or "manual-save"
        content = self.query_one("#run_editor", TextArea).text
        try:
            save_script_with_version(self.selected_script, content, note=note)
        except ValueError as exc:
            self.set_status(f"Script save failed: {exc}")
            return

        self.load_selected_script_into_editor()
        self.set_status("Script saved with version snapshot")

    def restore_editor_script(self) -> None:
        self._sync_selected_from_cursor()
        if self.selected_script is None:
            self.set_status("No script selected")
            return

        selected = self.query_one("#run_version_select", Select).value
        if not isinstance(selected, str) or not selected:
            self.set_status("No script version selected")
            return

        try:
            restore_script_version(self.selected_script, selected)
        except ValueError as exc:
            self.set_status(f"Script restore failed: {exc}")
            return

        self.load_selected_script_into_editor()
        self.set_status(f"Restored script from {selected}")

    async def run_script(self) -> None:
        if self.running_task and not self.running_task.done():
            self.set_status("A run/bench process is already active")
            return

        self._sync_selected_from_cursor()
        if self.selected_script is None:
            self.set_status("No script selected")
            return

        extra_args_raw = self.query_one("#run_extra_args", Input).value.strip()
        try:
            extra_args = shlex.split(extra_args_raw) if extra_args_raw else []
        except ValueError as exc:
            self.set_status(f"Invalid extra args: {exc}")
            return

        env: Optional[Dict[str, str]] = None
        binary_select = self.query_one("#run_binary_select", Select)
        selected_binary = binary_select.value
        if isinstance(selected_binary, str) and selected_binary:
            env = {**os.environ, "LLAMA_SERVER": selected_binary}
            self.set_status(f"Binary override: LLAMA_SERVER={selected_binary}")

        cmd = command_for_script(self.selected_script, extra_args)
        model_name = self.selected_model_name()
        self._current_job_name = model_name
        self.running_started_at = asyncio.get_running_loop().time()
        self.set_runtime_state(f"Current: {model_name} ({self.mode})", "Resources: starting...")
        self.post_message(RunPanel.JobStarted(model_name, datetime.now().strftime("%H:%M:%S"), self.mode))
        self.running_task = asyncio.create_task(self._stream_command(cmd, env=env))
        await self.running_task

    async def _resource_snapshot_for_group(self, pgid: int) -> str:
        proc = await asyncio.create_subprocess_exec(
            "ps",
            "-g",
            str(pgid),
            "-o",
            "pid=,pcpu=,rss=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        rows = [line.strip() for line in out.decode("utf-8", errors="replace").splitlines() if line.strip()]

        pids: List[int] = []
        cpu_total = 0.0
        rss_kib_total = 0
        for row in rows:
            parts = row.split()
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                cpu = float(parts[1])
                rss = int(parts[2])
            except ValueError:
                continue
            pids.append(pid)
            cpu_total += cpu
            rss_kib_total += rss

        gpu_mem_mib: Optional[int] = None
        if pids and shutil.which("nvidia-smi"):
            gpu = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            gpu_out, _ = await gpu.communicate()
            gpu_mem_mib = 0
            for row in gpu_out.decode("utf-8", errors="replace").splitlines():
                parts = [p.strip() for p in row.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    mem = int(parts[1])
                except ValueError:
                    continue
                if pid in pids:
                    gpu_mem_mib += mem

        elapsed = 0
        if self.running_started_at is not None:
            elapsed = int(asyncio.get_running_loop().time() - self.running_started_at)
        mins, secs = divmod(max(0, elapsed), 60)
        rss_mib = rss_kib_total / 1024.0
        gpu_text = f"{gpu_mem_mib} MiB" if gpu_mem_mib is not None else "n/a"
        return (
            f"Resources: procs={len(pids)} cpu={cpu_total:.1f}% "
            f"ram={rss_mib:.1f} MiB gpu={gpu_text} elapsed={mins:02d}:{secs:02d}"
        )

    async def _resource_loop(self, pgid: int) -> None:
        while self.running_proc is not None:
            try:
                snapshot = await self._resource_snapshot_for_group(pgid)
                self.query_one("#run_resources", Static).update(snapshot)
            except Exception:
                # keep resource loop non-fatal for process execution
                pass
            await asyncio.sleep(1)

    async def _stop_resource_loop(self) -> None:
        if self.resource_task and not self.resource_task.done():
            self.resource_task.cancel()
            try:
                await self.resource_task
            except asyncio.CancelledError:
                pass
        self.resource_task = None

    async def _stream_command(self, cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
        self.set_status(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        self.running_proc = proc
        self.resource_task = asyncio.create_task(self._resource_loop(proc.pid))
        assert proc.stdout is not None

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self.set_status(line.decode("utf-8", errors="replace").rstrip())

        rc = await proc.wait()
        await self._stop_resource_loop()
        self.running_proc = None
        elapsed_secs = 0.0
        if self.running_started_at is not None:
            elapsed_secs = asyncio.get_running_loop().time() - self.running_started_at
        self.running_started_at = None
        self.set_runtime_state("Current: idle", f"Resources: exited (code {rc})")
        elapsed_str = f"{elapsed_secs:.0f}s" if elapsed_secs < 120 else f"{elapsed_secs/60:.1f}m"
        self.post_message(RunPanel.JobFinished(self._current_job_name, elapsed_str, rc, self.mode))

    async def stop_script(self) -> None:
        proc = self.running_proc
        if proc is None:
            self.set_status("No active run/bench process")
            return

        self.set_status("Stopping process...")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        finally:
            self.running_proc = None
            await self._stop_resource_loop()
            self.running_started_at = None
            self.set_runtime_state("Current: idle", "Resources: stopped")
        self.set_status("Process stopped")

    @on(DataTable.RowHighlighted, "#run_scripts_table")
    def on_script_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        if 0 <= idx < len(self.filtered):
            self.selected_script = self.filtered[idx]
            self.load_selected_script_into_editor()

    @on(DataTable.RowSelected, "#run_scripts_table")
    def on_script_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        if 0 <= idx < len(self.filtered):
            self.selected_script = self.filtered[idx]
            self.load_selected_script_into_editor()

    @on(Select.Changed, "#run_mode")
    def on_mode_changed(self, event: Select.Changed) -> None:
        value = str(event.value or "run")
        if value not in {"run", "bench"}:
            return
        self.mode = value
        self.refresh_table()

    @on(Input.Changed, "#run_filter")
    def on_filter_changed(self, _: Input.Changed) -> None:
        self.refresh_table()

    @on(Button.Pressed, "#run_refresh")
    def on_refresh(self) -> None:
        self.refresh_script_inventory()

    @on(Button.Pressed, "#run_binary_scan")
    def on_binary_scan(self) -> None:
        self.refresh_binary_selector()

    @on(Button.Pressed, "#run_start")
    async def on_start(self) -> None:
        await self.run_script()

    @on(Button.Pressed, "#run_stop")
    async def on_stop(self) -> None:
        await self.stop_script()

    @on(Button.Pressed, "#run_edit_reload")
    def on_edit_reload(self) -> None:
        self._sync_selected_from_cursor()
        self.load_selected_script_into_editor()

    @on(Button.Pressed, "#run_edit_save")
    def on_edit_save(self) -> None:
        self.save_editor_script()

    @on(Button.Pressed, "#run_edit_restore")
    def on_edit_restore(self) -> None:
        self.restore_editor_script()



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_port_from_script(content: str) -> Optional[int]:
    """Extract --port N from a shell script string."""
    m = re.search(r"--port\s+(\d+)", content)
    return int(m.group(1)) if m else None


async def detect_llama_port() -> Optional[int]:
    """Probe running llama-server processes for their port via pgrep."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pgrep", "-fa", "llama-server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        text = out.decode("utf-8", errors="replace")
        m = re.search(r"--port\s+(\d+)", text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback: probe common ports
    for port in (8080, 8001, 8000, 8888):
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                r = await client.get(f"http://localhost:{port}/v1/models")
                if r.status_code == 200:
                    return port
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Chat Panel
# ---------------------------------------------------------------------------

class ChatPanel(Static):
    DEFAULT_PORT = 8080

    def __init__(self) -> None:
        super().__init__(id="chat_panel")
        self._history: List[Dict[str, str]] = []
        self._active_task: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            # Status bar
            with Horizontal(classes="row"):
                yield Static("◉ disconnected", id="chat_status")
                yield Static("", id="chat_model_label")
                yield Input(value=str(self.DEFAULT_PORT), placeholder="port", id="chat_port", classes="chat_port_input")
                yield Button("Detect", id="chat_detect")
                yield Button("Connect", id="chat_connect", variant="primary")
                yield Button("Clear (Ctrl+X)", id="chat_clear")

            # Params row
            with Horizontal(classes="row chat_params_row"):
                yield Input(placeholder="System prompt…", id="chat_system_prompt")
                yield Input(value="0.8", id="chat_temp", classes="chat_small_input")
                yield Label("Temp")
                yield Input(value="2048", id="chat_max_tokens", classes="chat_small_input")
                yield Label("Max Tokens")
                yield Checkbox("Thinking /think", id="chat_thinking", value=False)
                yield Button("Save", id="chat_save")

            # Conversation history
            yield RichLog(id="chat_log", wrap=True, markup=True)

            # Streaming preview (hidden while not streaming)
            yield Static("", id="chat_stream_preview")

            # Input row
            with Horizontal(classes="row chat_input_row"):
                yield Input(placeholder="Message… (Enter to send)", id="chat_input")
                yield Button("Send ↵", id="chat_send", variant="success")

            yield Label(
                "Enter send · Ctrl+X clear · Ctrl+G connect · Ctrl+B detect · Alt+S save chat · Thinking=Qwen3",
                classes="key_hint",
            )

    def on_mount(self) -> None:
        self.query_one("#chat_stream_preview", Static).display = False
        self.run_worker(self._auto_connect(), exclusive=False)

    async def _auto_connect(self) -> None:
        port_input = self.query_one("#chat_port", Input)
        port_str = port_input.value.strip()
        try:
            port = int(port_str) if port_str else self.DEFAULT_PORT
        except ValueError:
            port = self.DEFAULT_PORT

        model = await self._probe_model(port)
        if model:
            self._set_connected(port, model)
        else:
            self._set_disconnected()

    async def _probe_model(self, port: int) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"http://localhost:{port}/v1/models")
                if r.status_code == 200:
                    data = r.json()
                    models = data.get("data", [])
                    return models[0].get("id", "unknown") if models else "unknown"
        except Exception:
            pass
        return None

    def _set_connected(self, port: int, model: str) -> None:
        self.query_one("#chat_status", Static).update(f"[green]◉ localhost:{port}[/green]")
        self.query_one("#chat_model_label", Static).update(f"[dim]{model}[/dim]")
        self.query_one("#chat_port", Input).value = str(port)

    def _set_disconnected(self) -> None:
        self.query_one("#chat_status", Static).update("[red]◉ disconnected[/red]")
        self.query_one("#chat_model_label", Static).update("")

    def _current_port(self) -> int:
        try:
            return int(self.query_one("#chat_port", Input).value.strip())
        except ValueError:
            return self.DEFAULT_PORT

    def _log(self, text: str) -> None:
        self.query_one("#chat_log", RichLog).write(text)

    def focus_input(self) -> None:
        self.query_one("#chat_input", Input).focus()

    def clear_chat(self) -> None:
        self._history.clear()
        self.query_one("#chat_log", RichLog).clear()
        self._log("[dim]Chat cleared[/dim]")

    async def send_message(self) -> None:
        if self._active_task and not self._active_task.done():
            self._log("[yellow]⚠ Response in progress — please wait[/yellow]")
            return

        inp = self.query_one("#chat_input", Input)
        user_text = inp.value.strip()
        if not user_text:
            return
        inp.value = ""

        self._history.append({"role": "user", "content": user_text})
        self._log(f"[bold cyan]You:[/bold cyan] {user_text}")

        self._active_task = asyncio.create_task(self._stream_response())

    async def _stream_response(self) -> None:
        port = self._current_port()
        url = f"http://localhost:{port}/v1/chat/completions"

        # Build messages list with optional system prompt
        messages: List[Dict[str, str]] = []
        system_prompt = self.query_one("#chat_system_prompt", Input).value.strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._history)

        # Apply /think prefix if thinking checkbox is checked
        if self.query_one("#chat_thinking", Checkbox).value and messages:
            last = messages[-1]
            if last["role"] == "user":
                messages[-1] = {"role": "user", "content": f"/think\n{last['content']}"}

        # Read params
        try:
            temperature = float(self.query_one("#chat_temp", Input).value.strip())
        except ValueError:
            temperature = 0.8
        try:
            max_tokens = int(self.query_one("#chat_max_tokens", Input).value.strip())
        except ValueError:
            max_tokens = 2048

        payload = {
            "model": "default",
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }

        log = self.query_one("#chat_log", RichLog)
        preview = self.query_one("#chat_stream_preview", Static)
        full_reply = ""
        usage_info: Optional[Dict[str, Any]] = None
        start_time = asyncio.get_event_loop().time()

        preview.display = True
        preview.update("…")

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload,
                                         headers={"Accept": "text/event-stream"}) as resp:
                    if resp.status_code != 200:
                        self._log(f"[red]Server error {resp.status_code}[/red]")
                        preview.display = False
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # Capture usage from final chunk
                        if "usage" in chunk and chunk["usage"]:
                            usage_info = chunk["usage"]
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if not token:
                            continue
                        full_reply += token
                        preview.update(full_reply)

        except httpx.ConnectError:
            self._log(f"[red]Cannot connect to localhost:{port} — is the server running?[/red]")
            self._history.pop()  # remove the unanswered user message
            preview.display = False
            return
        except Exception as exc:
            self._log(f"[red]Error: {exc}[/red]")
            preview.display = False
            return

        preview.display = False

        if full_reply:
            elapsed = asyncio.get_event_loop().time() - start_time
            log.write(f"[bold green]Assistant:[/bold green] {full_reply}")
            if usage_info:
                comp_tokens = usage_info.get("completion_tokens", 0)
                tps = comp_tokens / elapsed if elapsed > 0 else 0.0
                log.write(f"[dim]  ↳ {comp_tokens} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/dim]")
            self._history.append({"role": "assistant", "content": full_reply})

    def save_chat(self) -> None:
        if not self._history:
            self._log("[yellow]Nothing to save — chat is empty[/yellow]")
            return
        chats_dir = Path.home() / ".l3ms" / "chats"
        chats_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = chats_dir / f"{timestamp}.md"
        lines = [f"# Chat - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        for msg in self._history:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"\n## {role}\n{msg['content']}\n")
        filepath.write_text("\n".join(lines), encoding="utf-8")
        self._log(f"[green]Chat saved to {filepath}[/green]")

    async def do_detect(self) -> None:
        self._log("[dim]Detecting running llama-server…[/dim]")
        port = await detect_llama_port()
        if port:
            model = await self._probe_model(port)
            self._set_connected(port, model or "unknown")
            self._log(f"[green]Detected server on port {port}[/green]")
        else:
            self._set_disconnected()
            self._log("[yellow]No running llama-server found[/yellow]")

    async def do_connect(self) -> None:
        port = self._current_port()
        model = await self._probe_model(port)
        if model:
            self._set_connected(port, model)
            self._log(f"[green]Connected to localhost:{port} — model: {model}[/green]")
        else:
            self._set_disconnected()
            self._log(f"[red]Could not connect to localhost:{port}[/red]")

    @on(Button.Pressed, "#chat_send")
    async def on_send(self) -> None:
        await self.send_message()

    @on(Button.Pressed, "#chat_detect")
    async def on_detect(self) -> None:
        await self.do_detect()

    @on(Button.Pressed, "#chat_connect")
    async def on_connect(self) -> None:
        await self.do_connect()

    @on(Button.Pressed, "#chat_clear")
    def on_clear(self) -> None:
        self.clear_chat()

    @on(Button.Pressed, "#chat_save")
    def on_save_chat(self) -> None:
        self.save_chat()

    @on(Input.Submitted, "#chat_input")
    async def on_input_submitted(self, _: Input.Submitted) -> None:
        await self.send_message()


class MaintenancePanel(Static):
    def __init__(self) -> None:
        super().__init__(id="maintenance_panel")
        self.scripts: List[Path] = []
        self.filtered: List[Path] = []
        self.selected_script: Optional[Path] = None
        self.running_proc: Optional[asyncio.subprocess.Process] = None
        self.running_task: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Button("Refresh", id="maint_refresh")
                yield Button("▶ Run (Ctrl+R)", id="maint_start", variant="success")
                yield Button("■ Stop (Ctrl+S)", id="maint_stop", variant="error")
                yield Static("idle", id="maint_status")
            with Horizontal(classes="row"):
                yield Input(placeholder="filter", id="maint_filter")
            with Horizontal(classes="row main"):
                with Vertical(classes="left"):
                    yield DataTable(id="maint_table")
                    yield RichLog(id="maint_log", wrap=True, markup=False)
                with Vertical(classes="right"):
                    yield Label("Script Editor")
                    yield Static("No script selected", id="maint_selected_path")
                    with Horizontal(classes="row"):
                        yield Button("Save", id="maint_edit_save", variant="success")
                        yield Button("Reload", id="maint_edit_reload")
                    yield TextArea("", id="maint_editor")
            yield Label("Keys: Ctrl+R run · Ctrl+S stop · Ctrl+L clear log", classes="key_hint")

    def on_mount(self) -> None:
        table = self.query_one("#maint_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "script")
        self.refresh_scripts()
        self.focus_table()

    def set_status(self, message: str) -> None:
        self.query_one("#maint_log", RichLog).write(message)

    def focus_table(self) -> None:
        self.query_one("#maint_table", DataTable).focus()

    def clear_log(self) -> None:
        self.query_one("#maint_log", RichLog).clear()
        self.set_status("Maintenance log cleared")

    def set_status_label(self, text: str) -> None:
        self.query_one("#maint_status", Static).update(text)

    def refresh_scripts(self) -> None:
        self.scripts = sorted([p for p in ROOT.glob(MAINTENANCE_SCRIPT_GLOB) if p.is_file()])
        filter_text = self.query_one("#maint_filter", Input).value.strip().lower()
        table = self.query_one("#maint_table", DataTable)
        table.clear()
        self.filtered = []
        for script in self.scripts:
            rel = script.relative_to(ROOT).as_posix()
            if filter_text and filter_text not in rel.lower():
                continue
            self.filtered.append(script)
            idx = len(self.filtered) - 1
            table.add_row(str(idx), rel, key=str(idx))

        if self.filtered:
            self.selected_script = self.filtered[0]
            table.move_cursor(row=0, column=0)
            self._load_script_into_editor(self.selected_script)
        else:
            self.selected_script = None
            self.query_one("#maint_selected_path", Static).update("No script selected")
            self.query_one("#maint_editor", TextArea).text = ""

    def _load_script_into_editor(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            self.set_status(f"Failed to load: {exc}")
            return
        rel = path.relative_to(ROOT).as_posix()
        self.query_one("#maint_selected_path", Static).update(rel)
        self.query_one("#maint_editor", TextArea).text = content

    def _save_editor_script(self) -> None:
        if self.selected_script is None:
            self.set_status("No script selected")
            return
        content = self.query_one("#maint_editor", TextArea).text
        try:
            self.selected_script.write_text(content, encoding="utf-8")
            self.set_status(f"Saved {self.selected_script.name}")
        except Exception as exc:
            self.set_status(f"Save failed: {exc}")

    async def run_script(self) -> None:
        if self.running_task and not self.running_task.done():
            self.set_status("A maintenance script is already running")
            return
        if self.selected_script is None:
            self.set_status("No script selected")
            return
        cmd = ["bash", str(self.selected_script)]
        self.set_status_label(f"running: {self.selected_script.name}")
        self.running_task = asyncio.create_task(self._stream_command(cmd))
        await self.running_task

    async def _stream_command(self, cmd: List[str]) -> None:
        self.set_status(f"$ {' '.join(shlex.quote(p) for p in cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.running_proc = proc
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self.set_status(line.decode("utf-8", errors="replace").rstrip())
        rc = await proc.wait()
        self.running_proc = None
        self.set_status_label("idle")
        self.set_status(f"Script exited with code {rc}")

    async def stop_script(self) -> None:
        proc = self.running_proc
        if proc is None:
            self.set_status("No active script")
            return
        self.set_status("Stopping script…")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        finally:
            self.running_proc = None
            self.set_status_label("idle")
        self.set_status("Script stopped")

    @on(DataTable.RowHighlighted, "#maint_table")
    def on_script_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        if 0 <= idx < len(self.filtered):
            self.selected_script = self.filtered[idx]
            self._load_script_into_editor(self.selected_script)

    @on(Input.Changed, "#maint_filter")
    def on_filter_changed(self, _: Input.Changed) -> None:
        self.refresh_scripts()

    @on(Button.Pressed, "#maint_refresh")
    def on_refresh(self) -> None:
        self.refresh_scripts()

    @on(Button.Pressed, "#maint_start")
    async def on_start(self) -> None:
        await self.run_script()

    @on(Button.Pressed, "#maint_stop")
    async def on_stop(self) -> None:
        await self.stop_script()

    @on(Button.Pressed, "#maint_edit_save")
    def on_edit_save(self) -> None:
        self._save_editor_script()

    @on(Button.Pressed, "#maint_edit_reload")
    def on_edit_reload(self) -> None:
        if self.selected_script:
            self._load_script_into_editor(self.selected_script)


JOBS_FILE = Path.home() / ".l3ms" / "jobs.json"
JOBS_MAX = 200


class JobsPanel(Static):
    def __init__(self) -> None:
        super().__init__(id="jobs_panel")
        self._jobs: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Label("Job History")
                yield Button("Clear", id="jobs_clear")
            yield DataTable(id="jobs_table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "script", "mode", "started", "elapsed", "exit")
        self.load_jobs()

    def load_jobs(self) -> None:
        """Load persisted job history from disk."""
        try:
            if JOBS_FILE.exists():
                self._jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._jobs = []
        self._refresh_table()

    def save_jobs(self) -> None:
        """Persist job history to disk, capped at JOBS_MAX entries."""
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = self._jobs[-JOBS_MAX:]
            JOBS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_job_started(self, name: str, started: str, mode: str = "") -> None:
        self._jobs.append({"name": name, "started": started, "elapsed": "-", "exit": "…", "mode": mode})
        self._refresh_table()
        self.save_jobs()

    def update_job_finished(self, name: str, elapsed: str, exit_code: int, mode: str = "") -> None:
        for job in reversed(self._jobs):
            if job["name"] == name and job["exit"] == "…":
                job["elapsed"] = elapsed
                job["exit"] = str(exit_code)
                job["mode"] = mode
                break
        self._refresh_table()
        self.save_jobs()

    def _refresh_table(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.clear()
        for i, job in enumerate(self._jobs):
            table.add_row(
                str(i), job["name"], job.get("mode", ""),
                job["started"], job["elapsed"], job["exit"],
                key=str(i)
            )

    @on(Button.Pressed, "#jobs_clear")
    def on_clear(self) -> None:
        self._jobs.clear()
        self.query_one("#jobs_table", DataTable).clear()
        self.save_jobs()


class PlaceholderPanel(Static):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label(self._title)
        yield Static("Planned for next feature commit.")


class MainScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="download", id="main_tabs"):
            with TabPane("Download", id="download"):
                yield DownloadPanel()
            with TabPane("Model Ops", id="run"):
                yield RunPanel()
            with TabPane("Chat", id="chat"):
                yield ChatPanel()
            with TabPane("Maintenance", id="maintenance"):
                yield MaintenancePanel()
            with TabPane("Settings", id="settings"):
                yield PlaceholderPanel("Toolkit settings")
            with TabPane("Jobs", id="jobs"):
                yield JobsPanel()
        yield Footer()


class L3MSApp(App[None]):
    TITLE = "L3MS"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f1", "tab_download", "Download"),
        ("f2", "tab_run", "Run"),
        ("f3", "tab_chat", "Chat"),
        ("f4", "tab_maintenance", "Maintenance"),
        ("f5", "tab_settings", "Settings"),
        ("f6", "tab_jobs", "Jobs"),
        ("ctrl+r", "run_start", "Run Script"),
        ("ctrl+s", "run_stop", "Stop Script"),
        ("ctrl+f", "run_focus_filter", "Run Filter"),
        ("ctrl+j", "run_focus_table", "Run Table"),
        ("ctrl+u", "run_focus_editor", "Run Editor"),
        ("ctrl+l", "run_clear_log", "Run Log Clear"),
        ("ctrl+m", "run_toggle_mode", "Run Mode"),
        ("ctrl+p", "run_save_script", "Run Save Script"),
        ("ctrl+g", "chat_connect", "Chat Connect"),
        ("ctrl+b", "chat_detect", "Chat Detect"),
        ("ctrl+x", "chat_clear", "Chat Clear"),
        ("alt+s", "chat_save", "Save Chat"),
        ("alt+t", "download_focus_table", "Download Table"),
        ("alt+i", "download_focus_editor", "Download Editor"),
        ("alt+y", "download_clear_log", "Download Log Clear"),
        ("alt+o", "download_load", "Download Load Config"),
        ("alt+w", "download_save", "Download Save Config"),
        ("alt+v", "download_validate", "Download Validate"),
        ("alt+n", "download_add", "Download Add Model"),
        ("alt+a", "download_apply", "Download Apply Edit"),
        ("alt+k", "download_delete", "Download Delete Model"),
        ("alt+d", "download_selected", "Download Selected"),
        ("alt+e", "download_enabled", "Download Enabled"),
    ]

    def on_mount(self) -> None:
        self.push_screen(MainScreen())

    def activate_tab(self, tab_id: str) -> None:
        if not self.screen:
            return
        try:
            tabs = self.screen.query_one("#main_tabs", TabbedContent)
        except Exception:
            return
        tabs.active = tab_id

    def get_run_panel(self) -> Optional[RunPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#run_panel", RunPanel)
        except Exception:
            return None

    def active_tab(self) -> Optional[str]:
        if not self.screen:
            return None
        try:
            tabs = self.screen.query_one("#main_tabs", TabbedContent)
        except Exception:
            return None
        return str(tabs.active) if tabs.active else None

    def get_download_panel(self) -> Optional[DownloadPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#download_panel", DownloadPanel)
        except Exception:
            return None

    def get_chat_panel(self) -> Optional[ChatPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#chat_panel", ChatPanel)
        except Exception:
            return None

    def get_maintenance_panel(self) -> Optional[MaintenancePanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#maintenance_panel", MaintenancePanel)
        except Exception:
            return None

    def get_jobs_panel(self) -> Optional[JobsPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#jobs_panel", JobsPanel)
        except Exception:
            return None

    def action_tab_download(self) -> None:
        self.activate_tab("download")

    def action_tab_run(self) -> None:
        self.activate_tab("run")

    def action_tab_chat(self) -> None:
        self.activate_tab("chat")
        panel = self.get_chat_panel()
        if panel:
            panel.focus_input()

    def action_tab_maintenance(self) -> None:
        self.activate_tab("maintenance")

    def action_tab_settings(self) -> None:
        self.activate_tab("settings")

    def action_tab_jobs(self) -> None:
        self.activate_tab("jobs")

    async def action_run_start(self) -> None:
        tab = self.active_tab()
        if tab == "maintenance":
            panel = self.get_maintenance_panel()
            if panel:
                await panel.run_script()
            return
        if tab != "run":
            return
        panel = self.get_run_panel()
        if panel:
            await panel.run_script()

    async def action_run_stop(self) -> None:
        tab = self.active_tab()
        if tab == "maintenance":
            panel = self.get_maintenance_panel()
            if panel:
                await panel.stop_script()
            return
        if tab != "run":
            return
        panel = self.get_run_panel()
        if panel:
            await panel.stop_script()

    def action_run_focus_filter(self) -> None:
        if self.active_tab() != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.focus_filter()

    def action_run_focus_table(self) -> None:
        if self.active_tab() != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.focus_table()

    def action_run_focus_editor(self) -> None:
        if self.active_tab() != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.focus_editor()

    def action_run_clear_log(self) -> None:
        tab = self.active_tab()
        if tab == "maintenance":
            panel = self.get_maintenance_panel()
            if panel:
                panel.clear_log()
            return
        if tab != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.clear_log()

    def action_run_toggle_mode(self) -> None:
        if self.active_tab() != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.toggle_mode()

    def action_run_save_script(self) -> None:
        if self.active_tab() != "run":
            return
        panel = self.get_run_panel()
        if panel:
            panel.save_editor_script()

    def action_download_focus_table(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.focus_table()

    def action_download_focus_editor(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.focus_editor()

    def action_download_clear_log(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.clear_log()

    def action_download_load(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.load_current_config()

    def action_download_save(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.save_current_config()

    def action_download_validate(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.validate_current_config()

    def action_download_add(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.add_model()

    def action_download_apply(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.apply_model_edit()

    def action_download_delete(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            panel.delete_selected_model()

    async def action_download_selected(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            await panel.download_selected_model()

    async def action_download_enabled(self) -> None:
        if self.active_tab() != "download":
            return
        panel = self.get_download_panel()
        if panel:
            await panel.download_enabled_models()

    # ------------------------------------------------------------------
    # Chat actions
    # ------------------------------------------------------------------

    async def action_chat_connect(self) -> None:
        if self.active_tab() != "chat":
            return
        panel = self.get_chat_panel()
        if panel:
            await panel.do_connect()

    async def action_chat_detect(self) -> None:
        if self.active_tab() != "chat":
            return
        panel = self.get_chat_panel()
        if panel:
            await panel.do_detect()

    def action_chat_clear(self) -> None:
        if self.active_tab() != "chat":
            return
        panel = self.get_chat_panel()
        if panel:
            panel.clear_chat()

    async def action_chat_save(self) -> None:
        if self.active_tab() != "chat":
            return
        panel = self.get_chat_panel()
        if panel:
            panel.save_chat()

    def on_run_panel_job_started(self, event: RunPanel.JobStarted) -> None:
        panel = self.get_jobs_panel()
        if panel:
            panel.add_job_started(event.name, event.started)

    def on_run_panel_job_finished(self, event: RunPanel.JobFinished) -> None:
        panel = self.get_jobs_panel()
        if panel:
            panel.update_job_finished(event.name, event.elapsed, event.exit_code)

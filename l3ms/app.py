from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
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
BENCH_SCRIPT_GLOB = "bench-models/bench-llama-cpp-*.sh"


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
                    yield Checkbox("preserve_existing", id="m_preserve", value=True)
                    yield Input(placeholder="max_workers (blank = null)", id="m_workers")

            yield Label("Activity Log")
            yield RichLog(id="activity_log", wrap=True, markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#models_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "enabled", "repo_id", "pattern", "local_dir")
        self.load_current_config()
        self.focus_table()

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
        self.load_model_into_editor(idx)

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
                "preserve_existing": self.query_one("#m_preserve", Checkbox).value,
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
        self.query_one("#m_preserve", Checkbox).value = bool(model.get("preserve_existing", True))
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
                "preserve_existing": True,
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

            cmd = ["python3", str(DOWNLOAD_SCRIPT), "--repo-id", str(repo_id)]
            local_dir = str(model.get("local_dir", "")).strip()
            if local_dir:
                cmd.extend(["--local-dir", local_dir])

            allow = model.get("allow_patterns") or []
            if allow:
                cmd.append("--allow-patterns")
                cmd.extend([str(x) for x in allow])

            ignore = model.get("ignore_patterns") or []
            if ignore:
                cmd.append("--ignore-patterns")
                cmd.extend([str(x) for x in ignore])

            revision = str(model.get("revision", "")).strip()
            if revision:
                cmd.extend(["--revision", revision])

            if model.get("force_download", False):
                cmd.append("--force-download")
            if not model.get("preserve_existing", True):
                cmd.append("--no-preserve-existing")

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

    @on(DataTable.RowSelected, "#models_table")
    def on_model_row(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self._set_selected_index(idx)

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
    def __init__(self) -> None:
        super().__init__(id="run_panel")
        self.mode = "run"
        self.run_scripts: List[Path] = []
        self.bench_scripts: List[Path] = []
        self.filtered: List[Path] = []
        self.selected_script: Optional[Path] = None
        self.running_proc: Optional[asyncio.subprocess.Process] = None
        self.running_task: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Label("Mode")
                yield Select([("Run", "run"), ("Bench", "bench")], value="run", id="run_mode")
                yield Button("Refresh", id="run_refresh")
                yield Button("Start (Ctrl+R)", id="run_start", variant="success")
                yield Button("Stop (Ctrl+S)", id="run_stop", variant="error")

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

    def refresh_script_inventory(self) -> None:
        self.run_scripts = collect_scripts(RUN_SCRIPT_GLOB)
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

        cmd = command_for_script(self.selected_script, extra_args)
        self.running_task = asyncio.create_task(self._stream_command(cmd))
        await self.running_task

    async def _stream_command(self, cmd: List[str]) -> None:
        self.set_status(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
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
        self.set_status(f"Process exited with code {rc}")

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
            with TabPane("Run Models", id="run"):
                yield RunPanel()
            with TabPane("Maintenance", id="maintenance"):
                yield PlaceholderPanel("Maintenance scripts")
            with TabPane("Settings", id="settings"):
                yield PlaceholderPanel("Toolkit settings")
            with TabPane("Jobs", id="jobs"):
                yield PlaceholderPanel("Job history and logs")
        yield Footer()


class L3MSApp(App[None]):
    TITLE = "L3MS"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f1", "tab_download", "Download Tab"),
        ("f2", "tab_run", "Run Tab"),
        ("f3", "tab_maintenance", "Maintenance Tab"),
        ("f4", "tab_settings", "Settings Tab"),
        ("f5", "tab_jobs", "Jobs Tab"),
        ("ctrl+r", "run_start", "Run Script"),
        ("ctrl+s", "run_stop", "Stop Script"),
        ("ctrl+f", "run_focus_filter", "Run Filter"),
        ("ctrl+j", "run_focus_table", "Run Table"),
        ("ctrl+u", "run_focus_editor", "Run Editor"),
        ("ctrl+l", "run_clear_log", "Run Log Clear"),
        ("ctrl+m", "run_toggle_mode", "Run Mode"),
        ("ctrl+p", "run_save_script", "Run Save Script"),
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

    def get_download_panel(self) -> Optional[DownloadPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#download_panel", DownloadPanel)
        except Exception:
            return None

    def action_tab_download(self) -> None:
        self.activate_tab("download")

    def action_tab_run(self) -> None:
        self.activate_tab("run")

    def action_tab_maintenance(self) -> None:
        self.activate_tab("maintenance")

    def action_tab_settings(self) -> None:
        self.activate_tab("settings")

    def action_tab_jobs(self) -> None:
        self.activate_tab("jobs")

    async def action_run_start(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            await panel.run_script()

    async def action_run_stop(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            await panel.stop_script()

    def action_run_focus_filter(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.focus_filter()

    def action_run_focus_table(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.focus_table()

    def action_run_focus_editor(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.focus_editor()

    def action_run_clear_log(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.clear_log()

    def action_run_toggle_mode(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.toggle_mode()

    def action_run_save_script(self) -> None:
        self.activate_tab("run")
        panel = self.get_run_panel()
        if panel:
            panel.save_editor_script()

    def action_download_focus_table(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.focus_table()

    def action_download_focus_editor(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.focus_editor()

    def action_download_clear_log(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.clear_log()

    def action_download_load(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.load_current_config()

    def action_download_save(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.save_current_config()

    def action_download_validate(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.validate_current_config()

    def action_download_add(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.add_model()

    def action_download_apply(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.apply_model_edit()

    def action_download_delete(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            panel.delete_selected_model()

    async def action_download_selected(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            await panel.download_selected_model()

    async def action_download_enabled(self) -> None:
        self.activate_tab("download")
        panel = self.get_download_panel()
        if panel:
            await panel.download_enabled_models()

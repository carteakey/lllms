from __future__ import annotations

import asyncio
import shlex
import subprocess
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

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = ROOT / "model_downloader" / "download_hf_model.py"


class DownloadPanel(Static):
    def __init__(self) -> None:
        super().__init__()
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

    def set_status(self, message: str) -> None:
        log = self.query_one("#activity_log", RichLog)
        log.write(message)

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
        options = [(name, name) for name in versions]
        select.set_options(options)

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
        model = self.model_from_editor()
        self.config["models"][self.selected_index] = model
        self.refresh_models_table()

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

    @on(DataTable.RowSelected, "#models_table")
    def on_model_row(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self.selected_index = idx
        self.load_model_into_editor(idx)

    @on(Button.Pressed, "#btn_load")
    def on_load(self) -> None:
        self.load_current_config()

    @on(Button.Pressed, "#btn_validate")
    def on_validate(self) -> None:
        if self.selected_index is not None:
            try:
                self.apply_editor_to_selected()
            except ValueError as exc:
                self.set_status(f"Validation error: {exc}")
                return

        self.config["base_models_dir"] = self.query_one("#base_models_dir", Input).value.strip()
        errors = validate_config(self.config)
        if errors:
            self.set_status("Validation failed:")
            for err in errors:
                self.set_status(f"- {err}")
            return
        self.set_status("Config validation passed")

    @on(Button.Pressed, "#btn_save")
    def on_save(self) -> None:
        try:
            if self.selected_index is not None:
                self.apply_editor_to_selected()
            self.config["base_models_dir"] = self.query_one("#base_models_dir", Input).value.strip()
            note = self.query_one("#save_note", Input).value.strip() or "manual-save"
            save_config(self.config_path, self.config, note=note)
            self._update_version_select()
            self.set_status(f"Saved config: {self.config_path}")
        except ValueError as exc:
            self.set_status(f"Save failed: {exc}")

    @on(Button.Pressed, "#btn_restore")
    def on_restore(self) -> None:
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

    @on(Button.Pressed, "#btn_add")
    def on_add(self) -> None:
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

    @on(Button.Pressed, "#btn_apply")
    def on_apply(self) -> None:
        try:
            self.apply_editor_to_selected()
            self.set_status("Applied editor changes to selected model")
        except ValueError as exc:
            self.set_status(f"Apply failed: {exc}")

    @on(Button.Pressed, "#btn_delete")
    def on_delete(self) -> None:
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

    @on(Button.Pressed, "#btn_download_selected")
    async def on_download_selected(self) -> None:
        if self.active_download and not self.active_download.done():
            self.set_status("A download is already running")
            return

        try:
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

    @on(Button.Pressed, "#btn_download_enabled")
    async def on_download_enabled(self) -> None:
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
        with TabbedContent(initial="download"):
            with TabPane("Download", id="download"):
                yield DownloadPanel()
            with TabPane("Run Models", id="run"):
                yield PlaceholderPanel("Run/bench scripts")
            with TabPane("Maintenance", id="maintenance"):
                yield PlaceholderPanel("Maintenance scripts")
            with TabPane("Settings", id="settings"):
                yield PlaceholderPanel("Toolkit settings")
            with TabPane("Jobs", id="jobs"):
                yield PlaceholderPanel("Job history and logs")
        yield Footer()


class HomelabTUI(App[None]):
    TITLE = "Homelab LLM Toolkit"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        self.push_screen(MainScreen())

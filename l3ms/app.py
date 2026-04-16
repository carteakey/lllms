from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shlex
import shutil
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.markup import escape as markup_escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
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
    command_for_script,
    list_script_versions,
    load_script,
    restore_script_version,
    save_script_with_version,
)
from . import llama_swap

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = ROOT / "model_downloader" / "download_hf_model.py"
BENCH_SCRIPT_GLOB = "bench-models/bench-llama-cpp-*.sh"
MAINTENANCE_SCRIPT_GLOB = "maintenance/*.sh"
LLAMA_SWAP_CONFIG_PATH = ROOT / "llama-swap.yaml"

_BYTES_PER_GB: int = 1_073_741_824
_ELAPSED_THRESHOLD_SECS: int = 120

_GGUF_MAGIC = b"GGUF"
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12
_MAX_GGUF_KV = 100_000
_MAX_GGUF_STRING_BYTES = 16 * 1024 * 1024
_MAX_GGUF_ARRAY_ITEMS = 50_000_000
_MAX_GGUF_TENSOR_DIMS = 64
_GGUF_CAPTURE_KEYS = {
    "general.name",
    "general.architecture",
    "general.file_type",
    "general.parameter_count",
    "general.basename",
    "general.size_label",
    "tokenizer.ggml.model",
}
_GGUF_FIXED_TYPE_SIZES = {
    _GGUF_TYPE_UINT8: 1,
    _GGUF_TYPE_INT8: 1,
    _GGUF_TYPE_UINT16: 2,
    _GGUF_TYPE_INT16: 2,
    _GGUF_TYPE_UINT32: 4,
    _GGUF_TYPE_INT32: 4,
    _GGUF_TYPE_FLOAT32: 4,
    _GGUF_TYPE_BOOL: 1,
    _GGUF_TYPE_UINT64: 8,
    _GGUF_TYPE_INT64: 8,
    _GGUF_TYPE_FLOAT64: 8,
}
_GGUF_FILE_TYPE_LABELS = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q2_K",
    10: "Q3_K_S",
    11: "Q3_K_M",
    12: "Q3_K_L",
    13: "Q4_K_S",
    14: "Q4_K_M",
    15: "Q5_K_S",
    16: "Q5_K_M",
    17: "Q6_K",
    18: "Q8_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "IQ3_XXS",
    22: "IQ1_S",
    23: "IQ4_NL",
    24: "IQ3_S",
    25: "IQ2_S",
    26: "IQ4_XS",
    27: "I8",
    28: "I16",
    29: "I32",
    30: "I64",
    31: "F64",
    32: "IQ1_M",
    33: "BF16",
    37: "TQ1_0",
    38: "TQ2_0",
}
_QUANT_TOKEN_RE = re.compile(
    r"(?i)(?:^|[-_.])((?:ud-)?(?:iq|q)\d(?:[_-][a-z0-9]+)*|(?:bf|fp|f)\d{2}|mxfp4)(?:[-_.]|$)"
)


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0 B"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return "0 B"


def _format_parameter_count(param_count: Optional[int]) -> str:
    if param_count is None or param_count <= 0:
        return "-"
    if param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.1f}B"
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    if param_count >= 1_000:
        return f"{param_count / 1_000:.1f}K"
    return str(param_count)


def _format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _read_exact(handle: Any, size: int) -> bytes:
    if size < 0:
        raise ValueError("negative read size")
    chunk = handle.read(size)
    if len(chunk) != size:
        raise ValueError("unexpected EOF while parsing GGUF")
    return chunk


def _read_u16(handle: Any) -> int:
    return struct.unpack("<H", _read_exact(handle, 2))[0]


def _read_u32(handle: Any) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: Any) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_i8(handle: Any) -> int:
    return struct.unpack("<b", _read_exact(handle, 1))[0]


def _read_i16(handle: Any) -> int:
    return struct.unpack("<h", _read_exact(handle, 2))[0]


def _read_i32(handle: Any) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_i64(handle: Any) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]


def _read_f32(handle: Any) -> float:
    return struct.unpack("<f", _read_exact(handle, 4))[0]


def _read_f64(handle: Any) -> float:
    return struct.unpack("<d", _read_exact(handle, 8))[0]


def _read_bool(handle: Any) -> bool:
    return struct.unpack("<?", _read_exact(handle, 1))[0]


def _read_gguf_string(handle: Any) -> str:
    size = _read_u64(handle)
    if size > _MAX_GGUF_STRING_BYTES:
        raise ValueError(f"GGUF string too large: {size} bytes")
    return _read_exact(handle, size).decode("utf-8", errors="replace")


def _skip_bytes(handle: Any, size: int) -> None:
    if size < 0:
        raise ValueError("negative seek size")
    handle.seek(size, os.SEEK_CUR)


def _skip_gguf_value(handle: Any, value_type: int) -> None:
    if value_type in _GGUF_FIXED_TYPE_SIZES:
        _skip_bytes(handle, _GGUF_FIXED_TYPE_SIZES[value_type])
        return

    if value_type == _GGUF_TYPE_STRING:
        _ = _read_gguf_string(handle)
        return

    if value_type == _GGUF_TYPE_ARRAY:
        nested_type = _read_u32(handle)
        nested_count = _read_u64(handle)
        if nested_count > _MAX_GGUF_ARRAY_ITEMS:
            raise ValueError(f"GGUF array too large: {nested_count} items")
        if nested_type in _GGUF_FIXED_TYPE_SIZES:
            _skip_bytes(handle, _GGUF_FIXED_TYPE_SIZES[nested_type] * nested_count)
            return
        for _ in range(nested_count):
            _skip_gguf_value(handle, nested_type)
        return

    raise ValueError(f"unsupported GGUF value type: {value_type}")


def _read_gguf_scalar_value(handle: Any, value_type: int) -> Any:
    if value_type == _GGUF_TYPE_UINT8:
        return _read_exact(handle, 1)[0]
    if value_type == _GGUF_TYPE_INT8:
        return _read_i8(handle)
    if value_type == _GGUF_TYPE_UINT16:
        return _read_u16(handle)
    if value_type == _GGUF_TYPE_INT16:
        return _read_i16(handle)
    if value_type == _GGUF_TYPE_UINT32:
        return _read_u32(handle)
    if value_type == _GGUF_TYPE_INT32:
        return _read_i32(handle)
    if value_type == _GGUF_TYPE_FLOAT32:
        return _read_f32(handle)
    if value_type == _GGUF_TYPE_BOOL:
        return _read_bool(handle)
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(handle)
    if value_type == _GGUF_TYPE_UINT64:
        return _read_u64(handle)
    if value_type == _GGUF_TYPE_INT64:
        return _read_i64(handle)
    if value_type == _GGUF_TYPE_FLOAT64:
        return _read_f64(handle)
    raise ValueError(f"unsupported GGUF scalar type: {value_type}")


def parse_gguf_metadata(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != _GGUF_MAGIC:
            raise ValueError("invalid GGUF magic")

        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version: {version}")

        tensor_count = _read_u64(handle)
        kv_count = _read_u64(handle)
        if kv_count > _MAX_GGUF_KV:
            raise ValueError(f"GGUF metadata key/value count too large: {kv_count}")

        metadata: Dict[str, Any] = {
            "gguf.version": version,
            "gguf.tensor_count": tensor_count,
            "gguf.kv_count": kv_count,
        }
        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            value_type = _read_u32(handle)
            if key in _GGUF_CAPTURE_KEYS and value_type != _GGUF_TYPE_ARRAY:
                metadata[key] = _read_gguf_scalar_value(handle, value_type)
            else:
                _skip_gguf_value(handle, value_type)

        derived_parameter_count = 0
        parsed_tensor_count = 0
        for _ in range(tensor_count):
            _ = _read_gguf_string(handle)  # tensor name
            n_dims = _read_u32(handle)
            if n_dims > _MAX_GGUF_TENSOR_DIMS:
                raise ValueError(f"GGUF tensor has too many dimensions: {n_dims}")

            element_count = 1
            for _ in range(n_dims):
                dim = _read_u64(handle)
                element_count *= max(1, dim)

            _ = _read_u32(handle)  # tensor type
            _ = _read_u64(handle)  # tensor data offset
            derived_parameter_count += element_count
            parsed_tensor_count += 1

        metadata["gguf.derived_parameter_count"] = derived_parameter_count
        metadata["gguf.parsed_tensor_count"] = parsed_tensor_count
        return metadata


def _guess_quantization_from_name(filename: str) -> Optional[str]:
    match = _QUANT_TOKEN_RE.search(filename)
    if not match:
        return None
    token = match.group(1).upper().replace("-", "_")
    if token.startswith("UD_"):
        token = token.replace("UD_", "UD-", 1)
    return token


def infer_quantization(filename: str, metadata: Dict[str, Any]) -> str:
    file_type = metadata.get("general.file_type")
    if isinstance(file_type, int):
        return _GGUF_FILE_TYPE_LABELS.get(file_type, f"ftype:{file_type}")
    guessed = _guess_quantization_from_name(filename)
    return guessed or "unknown"


def collect_scripts(pattern: str) -> List[Path]:
    return sorted([path for path in ROOT.glob(pattern) if path.is_file()])


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
                    yield Input(
                        placeholder="allow_patterns (comma separated)", id="m_allow"
                    )
                    yield Input(
                        placeholder="ignore_patterns (comma separated)", id="m_ignore"
                    )
                    yield Checkbox("force_download", id="m_force", value=False)
                    yield Input(
                        placeholder="max_workers (blank = null)", id="m_workers"
                    )

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
            free_gb = usage.free / _BYTES_PER_GB
            total_gb = usage.total / _BYTES_PER_GB
            label.update(
                f"💾 {free_gb:.0f} / {total_gb:.0f} GB free  {markup_escape(f'[{check}]')}"
            )
        except OSError:
            label.update(f"⚠ drive not mounted  {markup_escape(f'[{target}]')}")

    async def _estimate_download_size(
        self, repo_id: str, allow_patterns: List[str], ignore_patterns: List[str]
    ) -> Optional[int]:
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
                if allow_patterns and not any(
                    fnmatch.fnmatch(name, p) for p in allow_patterns
                ):
                    continue
                # Apply ignore_patterns filter
                if ignore_patterns and any(
                    fnmatch.fnmatch(name, p) for p in ignore_patterns
                ):
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
        if self.config.get("models") and 0 <= table.cursor_row < len(
            self.config["models"]
        ):
            self._set_selected_index(table.cursor_row)

    def load_current_config(self) -> None:
        path_value = self.query_one("#config_path", Input).value.strip()
        if path_value:
            self.config_path = Path(path_value).expanduser()

        self.config = load_config(self.config_path)
        self.query_one("#base_models_dir", Input).value = self.config.get(
            "base_models_dir", ""
        )
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
                raise ValueError(
                    "model max_workers must be a positive integer or blank"
                )
            workers = int(workers_raw)

        return normalize_model(
            {
                "enabled": self.query_one("#m_enabled", Checkbox).value,
                "repo_id": self.query_one("#m_repo_id", Input).value,
                "description": self.query_one("#m_description", Input).value,
                "local_dir": self.query_one("#m_local_dir", Input).value,
                "revision": self.query_one("#m_revision", Input).value,
                "allow_patterns": csv_to_list(self.query_one("#m_allow", Input).value),
                "ignore_patterns": csv_to_list(
                    self.query_one("#m_ignore", Input).value
                ),
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
        self.query_one("#m_description", Input).value = str(
            model.get("description", "")
        )
        self.query_one("#m_local_dir", Input).value = str(model.get("local_dir", ""))
        self.query_one("#m_revision", Input).value = str(model.get("revision", ""))
        self.query_one("#m_allow", Input).value = ", ".join(
            model.get("allow_patterns") or []
        )
        self.query_one("#m_ignore", Input).value = ", ".join(
            model.get("ignore_patterns") or []
        )
        self.query_one("#m_force", Checkbox).value = bool(
            model.get("force_download", False)
        )
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

        self.config["base_models_dir"] = self.query_one(
            "#base_models_dir", Input
        ).value.strip()
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
            self.config["base_models_dir"] = self.query_one(
                "#base_models_dir", Input
            ).value.strip()
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
                est_gb = est / _BYTES_PER_GB
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
        def __init__(
            self, name: str, started: str, mode: str, script_path: str = ""
        ) -> None:
            super().__init__()
            self.name = name
            self.started = started
            self.mode = mode
            self.script_path = script_path

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
        self.swap_models: List[llama_swap.SwapModel] = []
        self.bench_scripts: List[Path] = []
        self.filtered_scripts: List[Path] = []
        self.filtered_models: List[llama_swap.SwapModel] = []
        self.selected_script: Optional[Path] = None
        self.selected_model_id: Optional[str] = None
        # last model we successfully loaded via llama-swap; Stop targets this, not the cursor
        self.loaded_model_id: Optional[str] = None
        self.running_proc: Optional[asyncio.subprocess.Process] = None
        self.running_task: Optional[asyncio.Task[None]] = None
        self.resource_task: Optional[asyncio.Task[None]] = None
        self.running_started_at: Optional[float] = None
        self._current_job_name: str = "idle"
        self._swap_refresh_task: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Label("Mode")
                yield Select(
                    [("Run", "run"), ("Bench", "bench")], value="run", id="run_mode"
                )
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
                yield Input(
                    placeholder="extra args appended to script", id="run_extra_args"
                )

            with Horizontal(classes="row main"):
                with Vertical(classes="left"):
                    yield DataTable(id="run_scripts_table")
                    yield Label(
                        "Keys: Ctrl+F filter, Ctrl+J table, Ctrl+U editor, Ctrl+M toggle mode, "
                        "Ctrl+R start, Ctrl+S stop, Alt+P save script, Ctrl+L clear log"
                    )
                    yield RichLog(id="run_log", wrap=True, markup=False)

                with Vertical(classes="right"):
                    yield Label("Script Editor")
                    yield Static("No script selected", id="run_selected_path")
                    with Horizontal(classes="row"):
                        yield Select(
                            [], id="run_version_select", prompt="Script versions"
                        )
                        yield Input(placeholder="save note", id="run_save_note")
                    with Horizontal(classes="row"):
                        yield Button("Reload", id="run_edit_reload")
                        yield Button("Save", id="run_edit_save", variant="success")
                        yield Button("Restore", id="run_edit_restore")
                    yield TextArea("", id="run_editor")

    def on_mount(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        table.cursor_type = "row"
        self._install_table_columns()
        self.refresh_script_inventory()
        self.refresh_binary_selector()
        self.focus_table()

    def _install_table_columns(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        table.clear(columns=True)
        if self.mode == "run":
            table.add_columns("state", "model", "name")
        else:
            table.add_columns("#", "script")

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
        self.set_status(
            f"Found {len(binaries) - 1} llama-server binary/binaries in vendor/"
        )

    def refresh_script_inventory(self) -> None:
        self.bench_scripts = collect_scripts(BENCH_SCRIPT_GLOB)
        if self.mode == "run":
            self._schedule_swap_refresh()
        else:
            self.refresh_table()

    def _schedule_swap_refresh(self) -> None:
        if self._swap_refresh_task and not self._swap_refresh_task.done():
            return
        self._swap_refresh_task = asyncio.create_task(self._load_swap_models())

    async def _load_swap_models(self) -> None:
        try:
            models = await llama_swap.list_models()
        except Exception as exc:
            self.swap_models = []
            self.set_status(
                f"llama-swap unreachable at {llama_swap.DEFAULT_BASE_URL}: {exc}. "
                "Start llama-swap.service, then click Refresh."
            )
        else:
            self.swap_models = models
            self.set_status(f"Loaded {len(models)} model(s) from llama-swap")
        if self.mode == "run":
            self.refresh_table()

    def refresh_table(self) -> None:
        self._install_table_columns()
        table = self.query_one("#run_scripts_table", DataTable)
        filter_text = self.query_one("#run_filter", Input).value.strip().lower()

        if self.mode == "run":
            self.filtered_models = [
                m for m in self.swap_models
                if not filter_text or filter_text in m.id.lower() or filter_text in m.name.lower()
            ]
            for model in self.filtered_models:
                table.add_row(model.state, model.id, model.name, key=model.id)

            if self.filtered_models:
                first = self.filtered_models[0]
                self.selected_model_id = first.id
                table.move_cursor(row=0, column=0)
                self._show_model_details(first)
                self.set_status(
                    f"Loaded {len(self.filtered_models)} model(s) from llama-swap "
                    f"({len(self.swap_models)} total)"
                )
            else:
                self.selected_model_id = None
                self.query_one("#run_selected_path", Static).update(
                    "No model selected" if self.swap_models else "llama-swap unavailable — click Refresh"
                )
                self.query_one("#run_editor", TextArea).text = ""
                self.query_one("#run_version_select", Select).set_options([])
            return

        # bench mode: script-driven (unchanged)
        self.filtered_scripts = []
        for script in self.bench_scripts:
            rel = script.relative_to(ROOT).as_posix()
            if filter_text and filter_text not in rel.lower():
                continue
            self.filtered_scripts.append(script)
            idx = len(self.filtered_scripts) - 1
            table.add_row(str(idx), rel, key=str(idx))

        if self.filtered_scripts:
            self.selected_script = self.filtered_scripts[0]
            table.move_cursor(row=0, column=0)
            self.load_selected_script_into_editor()
            self.set_status(
                f"Loaded {len(self.filtered_scripts)} bench script(s) "
                f"({len(self.bench_scripts)} total before filter)"
            )
        else:
            self.selected_script = None
            self.query_one("#run_selected_path", Static).update("No script selected")
            self.query_one("#run_editor", TextArea).text = ""
            self.query_one("#run_version_select", Select).set_options([])
            self.set_status("No bench scripts match current filter")

    def _show_model_details(self, model: llama_swap.SwapModel) -> None:
        self.query_one("#run_selected_path", Static).update(f"{model.id}  —  state: {model.state}")
        details = [
            f"# llama-swap model: {model.id}",
            f"# name:   {model.name}" if model.name else "",
            f"# state:  {model.state}",
            f"# desc:   {model.description}" if model.description else "",
            "",
            "# Trigger load:",
            f"curl -X POST {llama_swap.DEFAULT_BASE_URL}/models/load \\",
            f"     -H 'Content-Type: application/json' \\",
            f"     -d '{{\"model\": \"{model.id}\"}}'",
            "",
            "# Chat:",
            f"curl {llama_swap.DEFAULT_BASE_URL}/v1/chat/completions \\",
            f"     -H 'Content-Type: application/json' \\",
            f"     -d '{{\"model\":\"{model.id}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'",
            "",
            "# Unload:",
            f"curl -X POST {llama_swap.DEFAULT_BASE_URL}/models/unload \\",
            f"     -H 'Content-Type: application/json' \\",
            f"     -d '{{\"model\": \"{model.id}\"}}'",
            "",
            "# Full definition lives in llama-swap.yaml.",
        ]
        self.query_one("#run_editor", TextArea).text = "\n".join(line for line in details if line is not None)
        self.query_one("#run_version_select", Select).set_options([])

    def toggle_mode(self) -> None:
        select = self.query_one("#run_mode", Select)
        self.mode = "bench" if self.mode == "run" else "run"
        select.value = self.mode
        self.refresh_script_inventory()

    def selected_model_name(self) -> str:
        if self.mode == "run":
            return self.selected_model_id or "idle"
        if self.selected_script is None:
            return "idle"
        name = self.selected_script.stem
        for prefix in (
            "bench-ik-llama-cpp-",
            "bench-llama-cpp-",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return name

    def _sync_selected_from_cursor(self) -> None:
        table = self.query_one("#run_scripts_table", DataTable)
        if self.mode == "run":
            if self.filtered_models and 0 <= table.cursor_row < len(self.filtered_models):
                self.selected_model_id = self.filtered_models[table.cursor_row].id
            return
        if self.filtered_scripts and 0 <= table.cursor_row < len(self.filtered_scripts):
            self.selected_script = self.filtered_scripts[table.cursor_row]

    def load_selected_script_into_editor(self) -> None:
        if self.mode == "run":
            # run mode: editor is a read-only detail pane populated by _show_model_details
            return
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
        if self.mode == "run":
            self.set_status(
                "Run mode is read-only; edit llama-swap.yaml to change a model entry"
            )
            return
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
        if self.mode == "run":
            self.set_status("Run mode is read-only; no script versions to restore")
            return
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

        if self.mode == "run":
            if not self.selected_model_id:
                self.set_status("No model selected")
                return
            model_id = self.selected_model_id
            self._current_job_name = model_id
            self.running_started_at = asyncio.get_running_loop().time()
            self.set_runtime_state(
                f"Current: {model_id} (run)", "Resources: loading via llama-swap..."
            )
            self.post_message(
                RunPanel.JobStarted(
                    model_id,
                    datetime.now().strftime("%H:%M:%S"),
                    self.mode,
                    # for run-mode, script_path carries the model ID so Jobs-tab retries work
                    script_path=model_id,
                )
            )
            self.running_task = asyncio.create_task(self._swap_load(model_id))
            await self.running_task
            return

        # bench mode: subprocess launch (unchanged)
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
        self.set_runtime_state(
            f"Current: {model_name} ({self.mode})", "Resources: starting..."
        )
        self.post_message(
            RunPanel.JobStarted(
                model_name,
                datetime.now().strftime("%H:%M:%S"),
                self.mode,
                script_path=str(self.selected_script) if self.selected_script else "",
            )
        )
        self.running_task = asyncio.create_task(self._stream_command(cmd, env=env))
        await self.running_task

    async def _swap_load(self, model_id: str) -> None:
        self.set_status(f"POST {llama_swap.DEFAULT_BASE_URL}/models/load  model={model_id}")
        try:
            result = await llama_swap.load_model(model_id)
        except Exception as exc:
            self.set_status(f"load failed: {exc}")
            self.set_runtime_state("Current: idle", "Resources: load failed")
            self.post_message(RunPanel.JobFinished(model_id, "0s", 1, self.mode))
            self.running_started_at = None
            return
        self.set_status(result)
        self.loaded_model_id = model_id
        await self._load_swap_models()
        await self._start_swap_resource_loop()
        elapsed_secs = 0.0
        if self.running_started_at is not None:
            elapsed_secs = asyncio.get_running_loop().time() - self.running_started_at
        self.running_started_at = None
        self.set_runtime_state(f"Current: {model_id} (loaded)", "Resources: polling...")
        elapsed_str = (
            f"{elapsed_secs:.0f}s"
            if elapsed_secs < _ELAPSED_THRESHOLD_SECS
            else f"{elapsed_secs / 60:.1f}m"
        )
        self.post_message(RunPanel.JobFinished(model_id, elapsed_str, 0, self.mode))

    async def _swap_unload(self, model_id: str) -> None:
        self.set_status(f"POST {llama_swap.DEFAULT_BASE_URL}/models/unload  model={model_id}")
        try:
            result = await llama_swap.unload_model(model_id)
        except Exception as exc:
            self.set_status(f"unload failed: {exc}")
            return
        self.set_status(result)
        if self.loaded_model_id == model_id:
            self.loaded_model_id = None
        await self._stop_resource_loop()
        await self._load_swap_models()

    async def _start_swap_resource_loop(self) -> None:
        await self._stop_resource_loop()
        pid = await _find_llama_swap_pid()
        if pid is None:
            return
        self.resource_task = asyncio.create_task(self._swap_resource_loop(pid))

    async def _swap_resource_loop(self, swap_pid: int) -> None:
        while self.loaded_model_id is not None:
            try:
                snapshot = await _resource_snapshot_for_ppid(swap_pid)
                self.query_one("#run_resources", Static).update(snapshot)
            except Exception:
                pass
            await asyncio.sleep(2)

    async def run_script_by_path(self, script_path: str, mode: str) -> None:
        """Select a script by full path and run it. Used by Jobs tab retry."""
        if mode == "run":
            # Run-mode retries are now model IDs, not paths. script_path holds the id.
            self.mode = "run"
            self.refresh_script_inventory()
            self.selected_model_id = script_path or None
            await self.run_script()
            return

        target = Path(script_path)
        if not target.exists():
            self.set_status(f"Script not found for retry: {script_path}")
            return
        if mode == "bench":
            self.mode = "bench"
            self.refresh_table()
        for i, s in enumerate(self.filtered_scripts):
            if s == target:
                self.selected_script = s
                table = self.query_one("#run_scripts_table", DataTable)
                table.move_cursor(row=i)
                self.load_selected_script_into_editor()
                break
        else:
            self.selected_script = target
            self.load_selected_script_into_editor()
        await self.run_script()

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
        rows = [
            line.strip()
            for line in out.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]

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

    async def _stream_command(
        self, cmd: List[str], env: Optional[Dict[str, str]] = None
    ) -> None:
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
        rc = -1
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self.set_status(line.decode("utf-8", errors="replace").rstrip())
            rc = await proc.wait()
        finally:
            await self._stop_resource_loop()
            self.running_proc = None
        elapsed_secs = 0.0
        if self.running_started_at is not None:
            elapsed_secs = asyncio.get_running_loop().time() - self.running_started_at
        self.running_started_at = None
        self.set_runtime_state("Current: idle", f"Resources: exited (code {rc})")
        elapsed_str = (
            f"{elapsed_secs:.0f}s"
            if elapsed_secs < _ELAPSED_THRESHOLD_SECS
            else f"{elapsed_secs / 60:.1f}m"
        )
        self.post_message(
            RunPanel.JobFinished(self._current_job_name, elapsed_str, rc, self.mode)
        )

    async def stop_script(self) -> None:
        if self.mode == "run":
            target = self.loaded_model_id
            if not target:
                self.set_status(
                    "No model is loaded via llama-swap; nothing to unload"
                )
                return
            await self._swap_unload(target)
            self.running_started_at = None
            self.set_runtime_state("Current: idle", "Resources: unloaded via llama-swap")
            return

        proc = self.running_proc
        if proc is None:
            self.set_status("No active bench process")
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

    def _handle_row_activated(self, event_key) -> None:
        key_str = str(event_key.value) if event_key is not None else ""
        if self.mode == "run":
            for model in self.filtered_models:
                if model.id == key_str:
                    self.selected_model_id = model.id
                    self._show_model_details(model)
                    return
            return
        try:
            idx = int(key_str)
        except (TypeError, ValueError):
            return
        if 0 <= idx < len(self.filtered_scripts):
            self.selected_script = self.filtered_scripts[idx]
            self.load_selected_script_into_editor()

    @on(DataTable.RowHighlighted, "#run_scripts_table")
    def on_script_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._handle_row_activated(event.row_key)

    @on(DataTable.RowSelected, "#run_scripts_table")
    def on_script_selected(self, event: DataTable.RowSelected) -> None:
        self._handle_row_activated(event.row_key)

    @on(Select.Changed, "#run_mode")
    def on_mode_changed(self, event: Select.Changed) -> None:
        value = str(event.value or "run")
        if value not in {"run", "bench"}:
            return
        if value == self.mode:
            return
        self.mode = value
        self.refresh_script_inventory()

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
# GGUF Model Browser Panel
# ---------------------------------------------------------------------------


class ModelBrowserPanel(Static):
    def __init__(self) -> None:
        super().__init__(id="model_browser_panel")
        self.rows: List[Dict[str, Any]] = []
        self.filtered_rows: List[Dict[str, Any]] = []
        self.selected_index: Optional[int] = None
        self.scan_task: Optional[asyncio.Task[List[Dict[str, Any]]]] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Input(placeholder="GGUF root directory", id="browser_root")
                yield Checkbox("recursive", value=True, id="browser_recursive")
                yield Button("Use Download Dir", id="browser_use_download_dir")
                yield Button("Scan (Alt+R)", id="browser_scan", variant="success")
            with Horizontal(classes="row"):
                yield Input(
                    placeholder="filter path / quant / architecture",
                    id="browser_filter",
                )
                yield Select(
                    [
                        ("Name ↑", "name_asc"),
                        ("Size ↓", "size_desc"),
                        ("Size ↑", "size_asc"),
                        ("Updated ↓", "mtime_desc"),
                        ("Updated ↑", "mtime_asc"),
                        ("Quant ↑", "quant_asc"),
                    ],
                    value="size_desc",
                    id="browser_sort",
                )
                yield Static("0 shown · 0 total · 0 B", id="browser_summary")
            with Horizontal(classes="row main"):
                with Vertical(classes="left"):
                    yield DataTable(id="browser_table")
                    yield Label(
                        "Keys: Alt+G path, Alt+J table, Alt+R scan",
                        classes="key_hint",
                    )
                with Vertical(classes="right"):
                    yield Static("No file selected", id="browser_selected_path")
                    yield Static(
                        "Scan a directory to inspect GGUF metadata.",
                        id="browser_details",
                    )
            yield RichLog(id="browser_log", wrap=True, markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#browser_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "gguf", "quant", "size", "params", "arch", "modified")
        self.query_one("#browser_root", Input).value = str(self._default_root_dir())
        self.set_status("Model browser ready")
        self.focus_table()
        self.run_worker(self.scan_models(), exclusive=False)

    def _default_root_dir(self) -> Path:
        configured = str(load_config(DEFAULT_CONFIG_PATH).get("base_models_dir", "")).strip()
        if configured:
            return Path(configured).expanduser()
        return ROOT

    def set_status(self, message: str) -> None:
        self.query_one("#browser_log", RichLog).write(message)

    def focus_path(self) -> None:
        self.query_one("#browser_root", Input).focus()

    def focus_table(self) -> None:
        self.query_one("#browser_table", DataTable).focus()

    def use_download_dir(self) -> None:
        root = self._default_root_dir()
        self.query_one("#browser_root", Input).value = str(root)
        self.set_status(f"Path set to {root}")

    async def scan_models(self) -> None:
        if self.scan_task and not self.scan_task.done():
            self.set_status("Scan already in progress")
            return

        raw_root = self.query_one("#browser_root", Input).value.strip()
        if not raw_root:
            self.set_status("Set a GGUF root directory first")
            return

        root = Path(raw_root).expanduser()
        if not root.exists():
            self.set_status(f"Path does not exist: {root}")
            return
        if not root.is_dir():
            self.set_status(f"Path is not a directory: {root}")
            return

        recursive = self.query_one("#browser_recursive", Checkbox).value
        mode = "recursive" if recursive else "top-level"
        self.query_one("#browser_summary", Static).update("Scanning…")
        self.set_status(f"Scanning {root} ({mode})")

        self.scan_task = asyncio.create_task(
            asyncio.to_thread(self._scan_directory, root, recursive)
        )
        try:
            self.rows = await self.scan_task
        except OSError as exc:
            self.rows = []
            self.filtered_rows = []
            self.selected_index = None
            self.query_one("#browser_summary", Static).update("Scan failed")
            self.set_status(f"Scan failed: {exc}")
            return
        finally:
            self.scan_task = None

        self.refresh_table()
        warning_count = sum(1 for row in self.rows if row.get("parse_error"))
        total_size = sum(row["size_bytes"] for row in self.rows)
        warning_text = f", {warning_count} warning(s)" if warning_count else ""
        self.set_status(
            f"Found {len(self.rows)} GGUF file(s), {_format_bytes(total_size)} total{warning_text}"
        )

    def _scan_directory(self, root: Path, recursive: bool) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        paths = self._iter_gguf_files(root, recursive)
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue

            metadata: Dict[str, Any] = {}
            parse_error = ""
            try:
                metadata = parse_gguf_metadata(path)
            except (OSError, ValueError, struct.error) as exc:
                parse_error = str(exc)

            try:
                display_path = path.relative_to(root).as_posix()
            except ValueError:
                display_path = path.as_posix()

            params_raw = metadata.get("general.parameter_count")
            params = params_raw if isinstance(params_raw, int) and params_raw > 0 else None
            arch_raw = metadata.get("general.architecture")
            arch = arch_raw if isinstance(arch_raw, str) and arch_raw else "-"
            name_raw = metadata.get("general.name")
            model_name = name_raw if isinstance(name_raw, str) else ""

            records.append(
                {
                    "path": path,
                    "display_path": display_path,
                    "size_bytes": int(stat.st_size),
                    "size_label": _format_bytes(int(stat.st_size)),
                    "mtime_ts": float(stat.st_mtime),
                    "modified_label": _format_mtime(stat.st_mtime),
                    "quant": infer_quantization(path.name, metadata),
                    "params": params,
                    "params_label": _format_parameter_count(params),
                    "arch": arch,
                    "model_name": model_name,
                    "metadata": metadata,
                    "parse_error": parse_error,
                }
            )
        return records

    def _iter_gguf_files(self, root: Path, recursive: bool) -> List[Path]:
        files: List[Path] = []
        if recursive:
            for dirpath, _, filenames in os.walk(root):
                base = Path(dirpath)
                for filename in filenames:
                    if filename.lower().endswith(".gguf"):
                        files.append(base / filename)
            return sorted(files)

        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".gguf":
                files.append(entry)
        return files

    def _sorted_rows(self, rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        if mode == "name_asc":
            return sorted(rows, key=lambda row: str(row["display_path"]).lower())
        if mode == "size_asc":
            return sorted(rows, key=lambda row: int(row["size_bytes"]))
        if mode == "mtime_desc":
            return sorted(rows, key=lambda row: float(row["mtime_ts"]), reverse=True)
        if mode == "mtime_asc":
            return sorted(rows, key=lambda row: float(row["mtime_ts"]))
        if mode == "quant_asc":
            return sorted(rows, key=lambda row: str(row["quant"]).lower())
        return sorted(rows, key=lambda row: int(row["size_bytes"]), reverse=True)

    def refresh_table(self) -> None:
        table = self.query_one("#browser_table", DataTable)
        table.clear()

        filter_text = self.query_one("#browser_filter", Input).value.strip().lower()
        sort_mode = str(self.query_one("#browser_sort", Select).value or "size_desc")
        rows = self.rows
        if filter_text:
            rows = [
                row
                for row in rows
                if filter_text in str(row["display_path"]).lower()
                or filter_text in str(row["quant"]).lower()
                or filter_text in str(row["arch"]).lower()
                or filter_text in str(row["model_name"]).lower()
            ]

        self.filtered_rows = self._sorted_rows(rows, sort_mode)
        for idx, row in enumerate(self.filtered_rows):
            table.add_row(
                str(idx),
                str(row["display_path"]),
                str(row["quant"]),
                str(row["size_label"]),
                str(row["params_label"]),
                str(row["arch"]),
                str(row["modified_label"]),
                key=str(idx),
            )

        shown_size = sum(int(row["size_bytes"]) for row in self.filtered_rows)
        summary = (
            f"{len(self.filtered_rows)} shown · {len(self.rows)} total · {_format_bytes(shown_size)} shown"
        )
        self.query_one("#browser_summary", Static).update(summary)

        if self.filtered_rows:
            table.move_cursor(row=0, column=0)
            self._set_selected_index(0)
        else:
            self.selected_index = None
            self.query_one("#browser_selected_path", Static).update("No file selected")
            self.query_one("#browser_details", Static).update(
                "No GGUF file matches current filter."
            )

    def _set_selected_index(self, idx: int) -> None:
        if not (0 <= idx < len(self.filtered_rows)):
            return
        self.selected_index = idx
        row = self.filtered_rows[idx]
        metadata = row["metadata"]

        self.query_one("#browser_selected_path", Static).update(
            markup_escape(str(row["path"]))
        )

        def _fmt(value: Any) -> str:
            if value is None:
                return "-"
            text = str(value).strip()
            if not text:
                return "-"
            return markup_escape(text)

        lines = [
            f"model: {_fmt(row['model_name'])}",
            f"quantization: {_fmt(row['quant'])}",
            f"size: {_fmt(row['size_label'])} ({int(row['size_bytes']):,} bytes)",
            f"architecture: {_fmt(row['arch'])}",
            f"parameters: {_fmt(row['params_label'])}",
            f"modified: {_fmt(row['modified_label'])}",
            f"gguf version: {_fmt(metadata.get('gguf.version'))}",
            f"tensor count: {_fmt(metadata.get('gguf.tensor_count'))}",
            f"file type id: {_fmt(metadata.get('general.file_type'))}",
            f"tokenizer: {_fmt(metadata.get('tokenizer.ggml.model'))}",
        ]
        parse_error = str(row.get("parse_error", "")).strip()
        if parse_error:
            lines.append(f"parse warning: {_fmt(parse_error)}")

        self.query_one("#browser_details", Static).update("\n".join(lines))

    @on(DataTable.RowHighlighted, "#browser_table")
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self._set_selected_index(idx)

    @on(DataTable.RowSelected, "#browser_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            return
        self._set_selected_index(idx)

    @on(Input.Changed, "#browser_filter")
    def on_filter_changed(self, _: Input.Changed) -> None:
        self.refresh_table()

    @on(Select.Changed, "#browser_sort")
    def on_sort_changed(self, _: Select.Changed) -> None:
        self.refresh_table()

    @on(Button.Pressed, "#browser_use_download_dir")
    def on_use_download_dir(self) -> None:
        self.use_download_dir()

    @on(Button.Pressed, "#browser_scan")
    async def on_scan_pressed(self) -> None:
        await self.scan_models()

    @on(Input.Submitted, "#browser_root")
    async def on_root_submitted(self, _: Input.Submitted) -> None:
        await self.scan_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_port_from_script(content: str) -> Optional[int]:
    """Extract --port N from a shell script string."""
    m = re.search(r"--port\s+(\d+)", content)
    return int(m.group(1)) if m else None


async def _find_llama_swap_pid() -> Optional[int]:
    """Return the PID of the llama-swap daemon, or None if not running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pgrep",
            "-f",
            "llama-swap",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except Exception:
        return None
    for line in out.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


async def _resource_snapshot_for_ppid(parent_pid: int) -> str:
    """ps-scrape CPU+RAM for the llama-swap children (the upstream llama-server procs)."""
    proc = await asyncio.create_subprocess_exec(
        "ps",
        "-o",
        "pid=,ppid=,pcpu=,rss=",
        "-ax",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    pids: List[int] = []
    cpu_total = 0.0
    rss_kib_total = 0
    for row in out.decode("utf-8", errors="replace").splitlines():
        parts = row.split()
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu = float(parts[2])
            rss = int(parts[3])
        except ValueError:
            continue
        if ppid != parent_pid:
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

    rss_mib = rss_kib_total / 1024.0
    gpu_text = f"{gpu_mem_mib} MiB" if gpu_mem_mib is not None else "n/a"
    return (
        f"Resources: upstreams={len(pids)} cpu={cpu_total:.1f}% "
        f"ram={rss_mib:.1f} MiB gpu={gpu_text}"
    )


async def detect_llama_port() -> Optional[int]:
    """Probe running llama-server processes for their port via pgrep."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pgrep",
            "-fa",
            "llama-server",
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
                yield Select([], id="chat_model_select", prompt="model")
                yield Input(
                    value=str(self.DEFAULT_PORT),
                    placeholder="port",
                    id="chat_port",
                    classes="chat_port_input",
                )
                yield Button("Detect", id="chat_detect")
                yield Button("Connect", id="chat_connect", variant="primary")
                yield Button("Clear (Ctrl+X)", id="chat_clear")
                yield Button("Sessions", id="chat_sessions")

            # Params row
            with Horizontal(classes="row chat_params_row"):
                yield Input(placeholder="System prompt…", id="chat_system_prompt")
                yield Input(value="0.8", id="chat_temp", classes="chat_small_input")
                yield Label("Temp")
                yield Input(
                    value="2048", id="chat_max_tokens", classes="chat_small_input"
                )
                yield Label("Max Tokens")
                yield Checkbox("Thinking /think", id="chat_thinking", value=False)
                yield Button("Save", id="chat_save")
                yield Button("Load", id="chat_load")

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

        models = await self._probe_models(port)
        if models:
            self._set_connected(port, models)
        else:
            self._set_disconnected()

    async def _probe_models(self, port: int) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"http://localhost:{port}/v1/models")
                if r.status_code == 200:
                    data = r.json()
                    return [
                        str(m.get("id"))
                        for m in data.get("data", [])
                        if isinstance(m, dict) and m.get("id")
                    ]
        except Exception:
            pass
        return []

    def _set_connected(self, port: int, models: List[str]) -> None:
        self.query_one("#chat_status", Static).update(
            f"[green]◉ localhost:{port}[/green]"
        )
        select = self.query_one("#chat_model_select", Select)
        options = [(m, m) for m in models] if models else []
        select.set_options(options)
        if options:
            current = select.value
            if not isinstance(current, str) or current not in models:
                select.value = models[0]
        self.query_one("#chat_port", Input).value = str(port)

    def _set_disconnected(self) -> None:
        self.query_one("#chat_status", Static).update("[red]◉ disconnected[/red]")
        self.query_one("#chat_model_select", Select).set_options([])

    def _current_model(self) -> str:
        value = self.query_one("#chat_model_select", Select).value
        if isinstance(value, str) and value:
            return value
        return "default"

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
            "model": self._current_model(),
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
                async with client.stream(
                    "POST", url, json=payload, headers={"Accept": "text/event-stream"}
                ) as resp:
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
            self._log(
                f"[red]Cannot connect to localhost:{port} — is the server running?[/red]"
            )
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
                log.write(
                    f"[dim]  ↳ {comp_tokens} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/dim]"
                )
            self._history.append({"role": "assistant", "content": full_reply})

    def save_chat(self) -> None:
        if not self._history:
            self._log("[yellow]Nothing to save — chat is empty[/yellow]")
            return
        chats_dir = Path.home() / ".l3ms" / "chats"
        chats_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save human-readable markdown
        md_path = chats_dir / f"{timestamp}.md"
        lines = [f"# Chat - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        for msg in self._history:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"\n## {role}\n{msg['content']}\n")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        # Save machine-readable JSON for session restore
        json_path = chats_dir / f"{timestamp}.json"
        json_path.write_text(
            json.dumps({"saved": timestamp, "history": self._history}, indent=2),
            encoding="utf-8",
        )
        self._log(f"[green]Chat saved → {md_path.name}[/green]")

    def load_chat_session(self, json_path: Path) -> None:
        """Restore a saved chat session from a JSON file."""
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            history = data.get("history", [])
            if not isinstance(history, list):
                raise ValueError("invalid history format")
        except Exception as exc:
            self._log(f"[red]Failed to load session: {exc}[/red]")
            return
        self._history = history
        log = self.query_one("#chat_log", RichLog)
        log.clear()
        log.write(f"[dim]── Loaded session: {json_path.stem} ──[/dim]")
        for msg in self._history:
            if msg.get("role") == "user":
                log.write(f"[bold cyan]You:[/bold cyan] {msg.get('content', '')}")
            else:
                log.write(
                    f"[bold green]Assistant:[/bold green] {msg.get('content', '')}"
                )
        self._log(f"[dim]── {len(self._history)} messages restored ──[/dim]")

    def open_sessions_browser(self) -> None:
        chats_dir = Path.home() / ".l3ms" / "chats"
        sessions = (
            sorted(chats_dir.glob("*.json"), reverse=True) if chats_dir.exists() else []
        )

        def on_result(path: Optional[Path]) -> None:
            if path:
                self.load_chat_session(path)

        self.app.push_screen(ChatHistoryScreen(sessions), callback=on_result)

    async def do_detect(self) -> None:
        self._log("[dim]Detecting running llama-server…[/dim]")
        port = await detect_llama_port()
        if port:
            models = await self._probe_models(port)
            self._set_connected(port, models)
            self._log(
                f"[green]Detected server on port {port} — {len(models)} model(s)[/green]"
            )
        else:
            self._set_disconnected()
            self._log("[yellow]No running llama-server found[/yellow]")

    async def do_connect(self) -> None:
        port = self._current_port()
        models = await self._probe_models(port)
        if models:
            self._set_connected(port, models)
            self._log(
                f"[green]Connected to localhost:{port} — {len(models)} model(s)[/green]"
            )
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

    @on(Button.Pressed, "#chat_sessions")
    def on_sessions(self) -> None:
        self.open_sessions_browser()

    @on(Button.Pressed, "#chat_load")
    def on_load_chat(self) -> None:
        self.open_sessions_browser()

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
            yield Label(
                "Keys: Ctrl+R run · Ctrl+S stop · Ctrl+L clear log", classes="key_hint"
            )

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
        self.scripts = sorted(
            [p for p in ROOT.glob(MAINTENANCE_SCRIPT_GLOB) if p.is_file()]
        )
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
        rc = -1
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self.set_status(line.decode("utf-8", errors="replace").rstrip())
            rc = await proc.wait()
        finally:
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


_L3MS_DATA_DIR = Path(os.environ.get("L3MS_DATA_DIR", str(Path.home() / ".l3ms")))
JOBS_FILE = _L3MS_DATA_DIR / "jobs.json"
JOBS_MAX = 200


class JobsPanel(Static):
    class StopRequest(Message):
        """Posted when the user asks to stop the currently running job."""

    class RetryRequest(Message):
        """Posted when the user asks to retry a selected finished job."""

        def __init__(self, script_path: str, mode: str) -> None:
            super().__init__()
            self.script_path = script_path
            self.mode = mode

    def __init__(self) -> None:
        super().__init__(id="jobs_panel")
        self._jobs: List[Dict[str, Any]] = []
        self._selected_idx: Optional[int] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel"):
            with Horizontal(classes="row"):
                yield Label("Job History")
                yield Button("■ Stop Running", id="jobs_stop", variant="error")
                yield Button("↺ Retry Selected", id="jobs_retry", variant="primary")
                yield Button("Clear", id="jobs_clear")
            yield DataTable(id="jobs_table")
            yield Label(
                "s stop running · r retry selected · Del clear history",
                classes="key_hint",
            )

    def on_mount(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "●", "script", "mode", "started", "elapsed", "exit")
        self.load_jobs()
        self._update_buttons()

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

    def add_job_started(
        self, name: str, started: str, mode: str = "", script_path: str = ""
    ) -> None:
        self._jobs.append(
            {
                "name": name,
                "started": started,
                "elapsed": "-",
                "exit": "…",
                "mode": mode,
                "script_path": script_path,
            }
        )
        self._selected_idx = len(self._jobs) - 1
        self._refresh_table()
        self._update_buttons()
        self.save_jobs()

    def update_job_finished(
        self, name: str, elapsed: str, exit_code: int, mode: str = ""
    ) -> None:
        for job in reversed(self._jobs):
            if job["name"] == name and job["exit"] == "…":
                job["elapsed"] = elapsed
                job["exit"] = str(exit_code)
                job["mode"] = mode
                break
        self._refresh_table()
        self._update_buttons()
        self.save_jobs()

    def _running_job_index(self) -> Optional[int]:
        """Return index of the currently running job (exit == '…'), or None."""
        for i, job in enumerate(self._jobs):
            if job.get("exit") == "…":
                return i
        return None

    def _update_buttons(self) -> None:
        running = self._running_job_index() is not None
        try:
            self.query_one("#jobs_stop", Button).disabled = not running
        except Exception:
            pass
        # Retry is available when a finished job with a known script_path is selected
        can_retry = False
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._jobs):
            job = self._jobs[self._selected_idx]
            can_retry = bool(job.get("script_path")) and job.get("exit") != "…"
        try:
            self.query_one("#jobs_retry", Button).disabled = not can_retry
        except Exception:
            pass

    def _refresh_table(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.clear()
        for i, job in enumerate(self._jobs):
            is_running = job.get("exit") == "…"
            indicator = "[green]▶[/green]" if is_running else ""
            table.add_row(
                str(i),
                indicator,
                job["name"],
                job.get("mode", ""),
                job["started"],
                job["elapsed"],
                job["exit"],
                key=str(i),
            )
        # Restore cursor
        if self._selected_idx is not None and self._selected_idx < len(self._jobs):
            try:
                table.move_cursor(row=self._selected_idx)
            except Exception:
                pass

    @on(DataTable.RowHighlighted, "#jobs_table")
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            self._selected_idx = int(str(event.row_key.value))
        except (TypeError, ValueError, AttributeError):
            pass
        self._update_buttons()

    @on(Button.Pressed, "#jobs_stop")
    def on_stop_btn(self) -> None:
        self.post_message(JobsPanel.StopRequest())

    @on(Button.Pressed, "#jobs_retry")
    def on_retry_btn(self) -> None:
        if self._selected_idx is None or self._selected_idx >= len(self._jobs):
            return
        job = self._jobs[self._selected_idx]
        script_path = job.get("script_path", "")
        mode = job.get("mode", "run")
        if script_path:
            self.post_message(JobsPanel.RetryRequest(script_path, mode))

    @on(Button.Pressed, "#jobs_clear")
    def on_clear(self) -> None:
        self._jobs.clear()
        self._selected_idx = None
        self.query_one("#jobs_table", DataTable).clear()
        self._update_buttons()
        self.save_jobs()


class PlaceholderPanel(Static):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label(self._title)
        yield Static("Planned for next feature commit.")


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------


def _build_help_content() -> str:
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "GLOBAL",
            [
                ("q", "Quit"),
                ("?", "This help screen"),
                ("Ctrl+P", "Command palette (all actions)"),
                ("F1", "→ Download tab"),
                ("F2", "→ Run / Model Ops tab"),
                ("F3", "→ Chat tab"),
                ("F4", "→ Maintenance tab"),
                ("F5", "→ Settings tab"),
                ("F6", "→ Jobs tab"),
                ("F7", "→ Model Browser tab"),
            ],
        ),
        (
            "MODEL BROWSER  (F7)",
            [
                ("Alt+R", "Scan GGUF files"),
                ("Alt+G", "Focus root path input"),
                ("Alt+J", "Focus GGUF table"),
                ("Enter", "Scan when root path input is focused"),
            ],
        ),
        (
            "JOBS  (F6)",
            [
                ("s", "Stop the running job"),
                ("r", "Retry selected finished job"),
                ("Del", "Clear all job history"),
            ],
        ),
        (
            "RUN / MODEL OPS  (F2)",
            [
                ("Ctrl+R", "Start script"),
                ("Ctrl+S", "Stop script"),
                ("Ctrl+M", "Toggle run ↔ bench mode"),
                ("Alt+P", "Save script"),
                ("Ctrl+F", "Focus filter input"),
                ("Ctrl+J", "Focus script table"),
                ("Ctrl+U", "Focus script editor"),
                ("Ctrl+L", "Clear log"),
            ],
        ),
        (
            "CHAT  (F3)",
            [
                ("Ctrl+G", "Connect to server"),
                ("Ctrl+B", "Auto-detect port"),
                ("Ctrl+X", "Clear chat history"),
                ("Alt+S", "Save chat session (md + json)"),
                ("Sessions", "Browse & load saved sessions"),
            ],
        ),
        (
            "DOWNLOAD  (F1)",
            [
                ("Alt+D", "Download selected model"),
                ("Alt+E", "Download all enabled models"),
                ("Alt+N", "Add new model entry"),
                ("Alt+A", "Apply editor → config"),
                ("Alt+K", "Delete selected model"),
                ("Alt+W", "Save config to disk"),
                ("Alt+O", "Reload config from disk"),
                ("Alt+V", "Validate config"),
                ("Alt+T", "Focus model table"),
                ("Alt+I", "Focus editor pane"),
                ("Alt+Y", "Clear log"),
            ],
        ),
    ]
    lines: list[str] = []
    for heading, bindings in sections:
        lines.append(f"\n [bold yellow]{heading}[/bold yellow]")
        for key, desc in bindings:
            lines.append(f"  [bold cyan]{key:<12}[/bold cyan] {desc}")
    lines.append("\n [dim]Esc or ? to close[/dim]")
    return "\n".join(lines)


class HelpScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("?", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help_dialog"):
            yield Static("⌨  L3MS Key Bindings", id="help_title")
            yield Static(_build_help_content(), id="help_body")


# ---------------------------------------------------------------------------
# Chat history browser
# ---------------------------------------------------------------------------


class ChatHistoryScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    def __init__(self, sessions: list) -> None:
        super().__init__()
        self._sessions: list = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="history_dialog"):
            yield Static("💬  Chat Sessions", id="history_title")
            if not self._sessions:
                yield Static(
                    "\n  [dim]No saved sessions found in ~/.l3ms/chats/[/dim]\n",
                    id="history_empty",
                )
            else:
                yield DataTable(id="history_table")
            yield Static(
                "[dim]  Enter to load · Esc to cancel[/dim]", id="history_hint"
            )

    def on_mount(self) -> None:
        if not self._sessions:
            return
        table = self.query_one("#history_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("saved", "msgs")
        for path in self._sessions:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                msg_count = str(len(data.get("history", [])))
                saved = data.get("saved", path.stem)
            except Exception:
                msg_count = "?"
                saved = path.stem
            table.add_row(saved, msg_count, key=str(path))
        table.focus()

    @on(DataTable.RowSelected, "#history_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(Path(str(event.row_key.value)))


# ---------------------------------------------------------------------------
# Command palette
# ---------------------------------------------------------------------------

PALETTE_COMMANDS: list = [
    ("→ Download tab", "tab_download"),
    ("→ Run / Model Ops tab", "tab_run"),
    ("→ Model Browser tab", "tab_browser"),
    ("→ Chat tab", "tab_chat"),
    ("→ Maintenance tab", "tab_maintenance"),
    ("→ Settings tab", "tab_settings"),
    ("→ Jobs tab", "tab_jobs"),
    ("Model Browser: Scan GGUF files", "browser_scan"),
    ("Model Browser: Focus root path", "browser_focus_path"),
    ("Model Browser: Focus table", "browser_focus_table"),
    ("Start script (Run)", "run_start"),
    ("Stop script (Run)", "run_stop"),
    ("Toggle run/bench mode", "run_toggle_mode"),
    ("Save script (Run)", "run_save_script"),
    ("Focus filter (Run)", "run_focus_filter"),
    ("Focus script table (Run)", "run_focus_table"),
    ("Focus script editor (Run)", "run_focus_editor"),
    ("Clear run log", "run_clear_log"),
    ("Chat: Connect to server", "chat_connect"),
    ("Chat: Auto-detect port", "chat_detect"),
    ("Chat: Clear history", "chat_clear"),
    ("Chat: Save session", "chat_save"),
    ("Download selected model", "download_selected"),
    ("Download all enabled models", "download_enabled"),
    ("Add new model entry", "download_add"),
    ("Apply model editor", "download_apply"),
    ("Delete selected model", "download_delete"),
    ("Save config to disk", "download_save"),
    ("Reload config from disk", "download_load"),
    ("Validate config", "download_validate"),
    ("Focus model table", "download_focus_table"),
    ("Focus download editor", "download_focus_editor"),
    ("Clear download log", "download_clear_log"),
    ("Show key bindings help", "show_help"),
    ("Quit", "quit"),
]


class CommandPaletteScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("ctrl+p", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="palette_dialog"):
            yield Static("⌘  Command Palette", id="palette_title")
            yield Input(placeholder="Type to filter commands…", id="palette_input")
            yield DataTable(id="palette_table", show_header=False)
            yield Static("[dim]  Enter to run · Esc to cancel[/dim]", id="palette_hint")

    def on_mount(self) -> None:
        table = self.query_one("#palette_table", DataTable)
        table.cursor_type = "row"
        table.add_column("command", width=54)
        self._populate(PALETTE_COMMANDS)
        self.query_one("#palette_input", Input).focus()

    def _populate(self, commands: list) -> None:
        table = self.query_one("#palette_table", DataTable)
        table.clear()
        for label, action in commands:
            table.add_row(label, key=action)

    @on(Input.Changed, "#palette_input")
    def on_filter(self, event: Input.Changed) -> None:
        q = event.value.lower()
        filtered = [(lbl, act) for lbl, act in PALETTE_COMMANDS if q in lbl.lower()]
        self._populate(filtered)

    @on(Input.Submitted, "#palette_input")
    def on_input_submitted(self, _: Input.Submitted) -> None:
        self._run_highlighted()

    @on(DataTable.RowSelected, "#palette_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(str(event.row_key.value))

    def _run_highlighted(self) -> None:
        table = self.query_one("#palette_table", DataTable)
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            action = str(row_key.value)
            if action:
                self.dismiss(action)
        except Exception:
            pass


# ---------------------------------------------------------------------------


class MainScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="download", id="main_tabs"):
            with TabPane("Download", id="download"):
                yield DownloadPanel()
            with TabPane("Model Ops", id="run"):
                yield RunPanel()
            with TabPane("Model Browser", id="browser"):
                yield ModelBrowserPanel()
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
        # ── always visible in footer ──────────────────────────────────
        Binding("q", "quit", "Quit", show=True),
        Binding("?", "show_help", "Help", show=True),
        Binding("f1", "tab_download", "Download", show=True),
        Binding("f2", "tab_run", "Run", show=True),
        Binding("f3", "tab_chat", "Chat", show=True),
        Binding("f4", "tab_maintenance", "Maint", show=True),
        Binding("f5", "tab_settings", "Settings", show=True),
        Binding("f6", "tab_jobs", "Jobs", show=True),
        Binding("f7", "tab_browser", "Browser", show=True),
        # ── Run / Model Ops (hidden – see ? Help) ────────────────────
        Binding("ctrl+r", "run_start", "Start Script", show=False),
        Binding("ctrl+s", "run_stop", "Stop Script", show=False),
        Binding("ctrl+f", "run_focus_filter", "Filter", show=False),
        Binding("ctrl+j", "run_focus_table", "Table", show=False),
        Binding("ctrl+u", "run_focus_editor", "Editor", show=False),
        Binding("ctrl+l", "run_clear_log", "Clear Log", show=False),
        Binding("ctrl+m", "run_toggle_mode", "Toggle Mode", show=False),
        Binding("alt+p", "run_save_script", "Save Script", show=False),
        # ── Command palette ───────────────────────────────────────────
        Binding("ctrl+p", "show_command_palette", "Palette", show=True),
        # ── Jobs (hidden) ─────────────────────────────────────────────
        Binding("s", "jobs_stop", "Stop Job", show=False),
        Binding("r", "jobs_retry", "Retry Job", show=False),
        # ── Chat (hidden) ─────────────────────────────────────────────
        Binding("ctrl+g", "chat_connect", "Connect", show=False),
        Binding("ctrl+b", "chat_detect", "Detect", show=False),
        Binding("ctrl+x", "chat_clear", "Clear Chat", show=False),
        Binding("alt+s", "chat_save", "Save Chat", show=False),
        # ── Download (hidden) ─────────────────────────────────────────
        Binding("alt+t", "download_focus_table", "Table", show=False),
        Binding("alt+i", "download_focus_editor", "Editor", show=False),
        Binding("alt+y", "download_clear_log", "Clear Log", show=False),
        Binding("alt+o", "download_load", "Load Config", show=False),
        Binding("alt+w", "download_save", "Save Config", show=False),
        Binding("alt+v", "download_validate", "Validate", show=False),
        Binding("alt+n", "download_add", "Add Model", show=False),
        Binding("alt+a", "download_apply", "Apply Edit", show=False),
        Binding("alt+k", "download_delete", "Delete Model", show=False),
        Binding("alt+d", "download_selected", "Download", show=False),
        Binding("alt+e", "download_enabled", "Dl Enabled", show=False),
        # ── Model Browser (hidden) ────────────────────────────────────
        Binding("alt+r", "browser_scan", "Scan GGUF", show=False),
        Binding("alt+g", "browser_focus_path", "Browser Path", show=False),
        Binding("alt+j", "browser_focus_table", "Browser Table", show=False),
    ]

    def on_mount(self) -> None:
        self.push_screen(MainScreen())

    async def action_quit(self) -> None:
        """Graceful shutdown: terminate running processes and cancel async tasks."""
        run_panel = self.get_run_panel()
        if run_panel:
            if run_panel.running_proc:
                try:
                    run_panel.running_proc.terminate()
                except Exception:
                    pass
            if run_panel.resource_task and not run_panel.resource_task.done():
                run_panel.resource_task.cancel()
            if run_panel.running_task and not run_panel.running_task.done():
                run_panel.running_task.cancel()

        maint_panel = self.get_maintenance_panel()
        if maint_panel:
            if maint_panel.running_proc:
                try:
                    maint_panel.running_proc.terminate()
                except Exception:
                    pass
            if maint_panel.running_task and not maint_panel.running_task.done():
                maint_panel.running_task.cancel()

        chat_panel = self.get_chat_panel()
        if chat_panel:
            if chat_panel._active_task and not chat_panel._active_task.done():
                chat_panel._active_task.cancel()

        browser_panel = self.get_model_browser_panel()
        if browser_panel:
            if browser_panel.scan_task and not browser_panel.scan_task.done():
                browser_panel.scan_task.cancel()

        self.exit()

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

    def get_model_browser_panel(self) -> Optional[ModelBrowserPanel]:
        if not self.screen:
            return None
        try:
            return self.screen.query_one("#model_browser_panel", ModelBrowserPanel)
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

    def action_tab_browser(self) -> None:
        self.activate_tab("browser")

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

    async def action_browser_scan(self) -> None:
        if self.active_tab() != "browser":
            return
        panel = self.get_model_browser_panel()
        if panel:
            await panel.scan_models()

    def action_browser_focus_path(self) -> None:
        if self.active_tab() != "browser":
            return
        panel = self.get_model_browser_panel()
        if panel:
            panel.focus_path()

    def action_browser_focus_table(self) -> None:
        if self.active_tab() != "browser":
            return
        panel = self.get_model_browser_panel()
        if panel:
            panel.focus_table()

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

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_show_command_palette(self) -> None:
        def handle_result(action: Optional[str]) -> None:
            if action:
                self.call_later(self.run_action, action)

        self.push_screen(CommandPaletteScreen(), callback=handle_result)

    def action_jobs_stop(self) -> None:
        if self.active_tab() != "jobs":
            return
        panel = self.get_jobs_panel()
        if panel:
            panel.post_message(JobsPanel.StopRequest())

    def action_jobs_retry(self) -> None:
        if self.active_tab() != "jobs":
            return
        panel = self.get_jobs_panel()
        if panel:
            panel.on_retry_btn()

    async def on_jobs_panel_stop_request(self, _: JobsPanel.StopRequest) -> None:
        panel = self.get_run_panel()
        if panel:
            await panel.stop_script()

    async def on_jobs_panel_retry_request(self, event: JobsPanel.RetryRequest) -> None:
        panel = self.get_run_panel()
        if panel:
            self.activate_tab("run")
            await panel.run_script_by_path(event.script_path, event.mode)

    def on_run_panel_job_started(self, event: RunPanel.JobStarted) -> None:
        panel = self.get_jobs_panel()
        if panel:
            panel.add_job_started(
                event.name,
                event.started,
                mode=event.mode,
                script_path=event.script_path,
            )

    def on_run_panel_job_finished(self, event: RunPanel.JobFinished) -> None:
        panel = self.get_jobs_panel()
        if panel:
            panel.update_job_finished(
                event.name, event.elapsed, event.exit_code, mode=event.mode
            )

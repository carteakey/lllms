"""Client helpers for the llama-swap HTTP API.

llama-swap is the single source of truth for servable models at runtime.
/v1/models omits `unlisted` entries by default; use /api/models (with
`all=true` if needed) for admin views. Load/unload are POST endpoints on
/models/load and /models/unload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:8080")
REQUEST_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


@dataclass
class SwapModel:
    id: str
    state: str  # "loaded" | "loading" | "unloaded" | "unknown"
    name: str = ""
    description: str = ""


def _normalize_state(entry: dict) -> str:
    raw = entry.get("state") or entry.get("status")
    if isinstance(raw, str) and raw:
        return raw
    if entry.get("loaded") is True:
        return "loaded"
    return "unknown"


async def list_models(base_url: str = DEFAULT_BASE_URL) -> List[SwapModel]:
    """Return every servable model. Raises httpx errors on connection failure."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
    models = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not model_id:
            continue
        models.append(
            SwapModel(
                id=str(model_id),
                state=_normalize_state(entry),
                name=str(entry.get("name") or ""),
                description=str(entry.get("description") or ""),
            )
        )
    models.sort(key=lambda m: m.id)
    return models


async def load_model(model_id: str, base_url: str = DEFAULT_BASE_URL) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=2.0)) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/models/load",
            json={"model": model_id},
        )
    return f"HTTP {resp.status_code} {resp.text.strip()[:200]}"


async def unload_model(model_id: str, base_url: str = DEFAULT_BASE_URL) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/models/unload",
            json={"model": model_id},
        )
    return f"HTTP {resp.status_code} {resp.text.strip()[:200]}"


async def probe(base_url: str = DEFAULT_BASE_URL) -> Optional[str]:
    """Return an error string if llama-swap is unreachable, else None."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
            resp.raise_for_status()
        return None
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"

"""Content-addressed manifests for preregistered experiment cells."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_state(repo: Optional[Path]) -> Dict[str, Any]:
    if repo is None:
        return {"commit": None, "dirty": None}
    repo = repo.resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def freeze_manifest(config: Any, path: Path, repo: Optional[Path] = None) -> Dict[str, Any]:
    """Write a manifest once; refuse to overwrite it with different content."""
    payload = _jsonable(config)
    manifest = {
        "schema_version": 1,
        "config": payload,
        "config_sha256": content_hash(payload),
        "git": git_state(repo),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("config_sha256") != manifest["config_sha256"]:
            raise FileExistsError(f"refusing to replace frozen manifest with new config: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

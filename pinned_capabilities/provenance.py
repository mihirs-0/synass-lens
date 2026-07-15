"""Content bindings and safe metadata inspection for experiment inputs.

The helpers in this module deliberately return JSON-compatible dictionaries so
they can be embedded directly in a frozen experiment manifest.  Snapshot
inspection never falls back to unrestricted pickle loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, Dict, Mapping, Optional

import numpy as np
import torch

from .manifest import content_hash


_HASH_CHUNK_BYTES = 1024 * 1024


def _regular_file_size(stream: BinaryIO, path: Path) -> int:
    details = os.fstat(stream.fileno())
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"provenance input is not a regular file: {path}")
    return int(details.st_size)


def _hash_stream(stream: BinaryIO, *, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    if chunk_size <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def sha256_file(path: Path, *, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 digest of a regular file without loading it at once."""
    path = Path(path)
    with path.open("rb") as stream:
        _regular_file_size(stream, path)
        return _hash_stream(stream, chunk_size=chunk_size)


def bind_file(path: Path) -> Dict[str, Any]:
    """Return a manifest-ready path, size, and content digest binding."""
    path = Path(path)
    with path.open("rb") as stream:
        size = _regular_file_size(stream, path)
        digest = _hash_stream(stream)
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def bind_reference(path: Path) -> Dict[str, Any]:
    """Bind a reference JSON file and verify that it contains a JSON object."""
    path = Path(path)
    with path.open("rb") as stream:
        size = _regular_file_size(stream, path)
        digest = _hash_stream(stream)
        stream.seek(0)
        try:
            payload = json.loads(stream.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"reference is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"reference must contain a JSON object: {path}")
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def bind_manifest(path: Path) -> Dict[str, Any]:
    """Bind a suite manifest and verify its internal configuration digest."""
    path = Path(path)
    with path.open("rb") as stream:
        size = _regular_file_size(stream, path)
        raw = stream.read()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid suite manifest structure: {path}")
    declared = payload.get("config_sha256")
    if not isinstance(declared, str):
        raise ValueError(f"suite manifest has no configuration digest: {path}")
    declared = _normalized_sha256(declared, label="manifest config hash")
    observed = content_hash(payload["config"])
    if declared != observed:
        raise ValueError(f"suite manifest configuration digest mismatch: {path}")
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "config_sha256": declared,
    }


def bind_nearest_manifest(input_path: Path, *, maximum_parent_hops: int = 2) -> Dict[str, Any]:
    """Bind the nearest ancestor manifest that declares an input artifact."""
    if maximum_parent_hops < 0:
        raise ValueError("maximum parent hops must be nonnegative")
    directory = Path(input_path).parent
    for _ in range(maximum_parent_hops + 1):
        candidate = directory / "manifest.json"
        if candidate.exists():
            return bind_manifest(candidate)
        directory = directory.parent
    raise FileNotFoundError(f"no source manifest found for input: {input_path}")


def _safe_snapshot_load(stream: BinaryIO) -> Mapping[str, Any]:
    """Load a checkpoint with a small allowlist and no arbitrary-pickle path."""
    numpy_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    try:
        with torch.serialization.safe_globals(numpy_globals):
            payload = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("snapshot cannot be inspected with restricted loading") from error
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot payload must be a mapping")
    return payload


def _json_metadata(value: Any, *, location: str = "metadata") -> Any:
    """Copy metadata while rejecting values that cannot enter a JSON manifest."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} contains a non-finite number")
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string key")
            result[key] = _json_metadata(item, location=f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_metadata(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{location} contains unsupported value {type(value).__name__}")


def _snapshot_info(payload: Mapping[str, Any]) -> Dict[str, Any]:
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, numbers.Integral):
        raise ValueError("snapshot schema_version must be an integer")
    if int(schema_version) != 1:
        raise ValueError(f"unsupported snapshot schema: {schema_version}")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, numbers.Integral) or int(step) < 0:
        raise ValueError("snapshot step must be a nonnegative integer")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("snapshot metadata must be a mapping")
    return {
        "schema_version": int(schema_version),
        "step": int(step),
        "metadata": _json_metadata(metadata),
    }


def inspect_snapshot(path: Path) -> Dict[str, Any]:
    """Return trusted structural fields using restricted checkpoint loading."""
    path = Path(path)
    with path.open("rb") as stream:
        _regular_file_size(stream, path)
        return _snapshot_info(_safe_snapshot_load(stream))


def _normalized_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return normalized


def _expected_config_hash(expected_config: Any, expected_config_sha256: Optional[str]) -> Optional[str]:
    derived = content_hash(expected_config) if expected_config is not None else None
    supplied = (
        _normalized_sha256(expected_config_sha256, label="expected config hash")
        if expected_config_sha256 is not None
        else None
    )
    if derived is not None and supplied is not None and derived != supplied:
        raise ValueError("expected config and expected config hash disagree")
    return supplied or derived


def _observed_config_hash(metadata: Mapping[str, Any]) -> Optional[str]:
    declared = metadata.get("config_sha256")
    declared_hash = (
        _normalized_sha256(declared, label="snapshot config hash")
        if declared is not None and isinstance(declared, str)
        else None
    )
    if declared is not None and not isinstance(declared, str):
        raise ValueError("snapshot config hash must be a string")
    config_key = next((key for key in ("config", "experiment_config") if key in metadata), None)
    embedded_hash = content_hash(metadata[config_key]) if config_key is not None else None
    if declared_hash is not None and embedded_hash is not None and declared_hash != embedded_hash:
        raise ValueError("snapshot embedded config and config hash disagree")
    return declared_hash or embedded_hash


def validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_seed: Optional[int] = None,
    expected_step: Optional[int] = None,
    expected_config: Any = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, str]:
    """Validate expected identity fields and report which checks were possible.

    Seed and configuration checks are marked ``unavailable`` when older
    snapshots do not carry the corresponding metadata.  Step is structural and
    therefore always available in a valid snapshot.
    """
    metadata = snapshot.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("snapshot metadata must be a mapping")
    checks: Dict[str, str] = {}

    if expected_seed is None:
        checks["seed"] = "not_requested"
    elif "seed" not in metadata:
        checks["seed"] = "unavailable"
    else:
        observed_seed = metadata["seed"]
        if isinstance(observed_seed, bool) or not isinstance(observed_seed, numbers.Integral):
            raise ValueError("snapshot seed metadata must be an integer")
        if int(observed_seed) != int(expected_seed):
            raise ValueError(
                f"snapshot seed mismatch: expected {expected_seed}, observed {observed_seed}"
            )
        checks["seed"] = "matched"

    if expected_step is None:
        checks["step"] = "not_requested"
    else:
        observed_step = snapshot.get("step")
        if isinstance(observed_step, bool) or not isinstance(observed_step, numbers.Integral):
            raise ValueError("snapshot step must be an integer")
        if int(observed_step) != int(expected_step):
            raise ValueError(
                f"snapshot step mismatch: expected {expected_step}, observed {observed_step}"
            )
        checks["step"] = "matched"

    expected_hash = _expected_config_hash(expected_config, expected_config_sha256)
    observed_hash = _observed_config_hash(metadata)
    if expected_hash is None:
        checks["config"] = "not_requested"
    elif observed_hash is None:
        checks["config"] = "unavailable"
    elif observed_hash != expected_hash:
        raise ValueError(
            f"snapshot config mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    else:
        checks["config"] = "matched"
    return checks


def bind_snapshot(
    path: Path,
    *,
    expected_seed: Optional[int] = None,
    expected_step: Optional[int] = None,
    expected_config: Any = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind snapshot bytes and validate their identity in one manifest record."""
    path = Path(path)
    with path.open("rb") as stream:
        size = _regular_file_size(stream, path)
        digest = _hash_stream(stream)
        stream.seek(0)
        snapshot = _snapshot_info(_safe_snapshot_load(stream))
    checks = validate_snapshot(
        snapshot,
        expected_seed=expected_seed,
        expected_step=expected_step,
        expected_config=expected_config,
        expected_config_sha256=expected_config_sha256,
    )
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "snapshot": snapshot,
        "checks": checks,
    }

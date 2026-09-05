"""Manifiestos pequeños para reconstruir experimentos y ejecuciones."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_manifest(
    *,
    name: str,
    configuration: Mapping[str, Any],
    data_files: Mapping[str, str] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Captura configuración y hashes; nunca inserta el contenido de los datos."""

    canonical = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    versions = {}
    for package in ("miopia", "numpy", "pandas", "scikit-learn", "spacy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": "1.0",
        "name": name,
        "run_id": f"{name}-{config_hash[:12]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(configuration),
        "configuration_sha256": config_hash,
        "data_sha256": {
            logical_name: sha256_file(path)
            for logical_name, path in sorted((data_files or {}).items())
        },
        "seed": seed,
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "packages": versions,
    }

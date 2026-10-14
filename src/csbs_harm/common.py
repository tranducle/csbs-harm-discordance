"""Shared paths and file utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.environ.get("CSBS_HARM_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = REPO_ROOT / "configs"
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

_HASH_BLOCK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_prefix: str, label: str) -> str:
    """Check a file against an expected SHA-256 prefix.

    Args:
        path: File to hash.
        expected_prefix: Leading hex characters (or the full digest) the file must match.
        label: Human-readable name used in the error message.

    Returns:
        The full digest of the file.

    Raises:
        ValueError: If the digest does not start with ``expected_prefix``.
    """
    actual = sha256_file(path)
    if not actual.startswith(expected_prefix.lower()):
        raise ValueError(
            f"{label} does not match the reference checksum.\n"
            f"  file:     {path}\n  expected: {expected_prefix}\n  actual:   {actual}\n"
            "Rebuild the file with scripts/prepare_csbs_data.py from the original survey releases."
        )
    return actual


def relative_to_repo(path: Path) -> str:
    """Return a repository-relative POSIX path.

    Symlinks are not followed, so raw data linked from elsewhere keeps its
    in-repository path. A path outside the repository is reduced to its file
    name so that no machine-specific location is recorded.
    """
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path(os.path.abspath(REPO_ROOT))).as_posix()
    except ValueError:
        return absolute.name


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON document with sorted keys, two-space indent and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", relative_to_repo(path))


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))

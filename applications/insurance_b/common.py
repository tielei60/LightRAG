"""Small file helpers for immutable experiment artifacts."""

import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def digest(value: bytes | str) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n"
        )


def inside(root: Path, path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the B application directory: {path}")
    return candidate


@contextmanager
def workspace_lock(root: Path, workspace: str, *, exclusive=False):
    if not workspace.startswith("insurance_b"):
        raise ValueError("A B workspace lock cannot name another application")
    path = inside(root, root / ".runtime/locks" / (workspace + ".lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(
                stream.fileno(),
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another B process owns this workspace; close its writer or server first"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

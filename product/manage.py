"""Local process and first-batch operations using the unmodified official API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "first3"
PID_FILE = OUTPUT / "server.pid"
MANIFEST = OUTPUT / "manifest.json"
RECEIPTS = OUTPUT / "uploads.json"


def settings() -> dict[str, str | None]:
    return dict(dotenv_values(ROOT / ".env", interpolate=False))


def base_url() -> str:
    config = settings()
    if config.get("HOST") != "127.0.0.1":
        raise ValueError("The local operator only supports the loopback host")
    return f"http://127.0.0.1:{int(config['PORT'])}"


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checked_files() -> list[tuple[Path, dict]]:
    manifest = json.loads(MANIFEST.read_text())
    source = Path(manifest["source_root"]).resolve(strict=True)
    documents = manifest["documents"]
    if len({entry["product"] for entry in documents}) != 3:
        raise ValueError("This batch must contain exactly three products")
    files = []
    names: set[str] = set()
    for entry in documents:
        path = (source / entry["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Source path escapes the selected source directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source changed: {path.name}")
        if path.name in names:
            raise ValueError("Duplicate upload filenames are not allowed")
        names.add(path.name)
        files.append((path, entry))
    return files


def start() -> None:
    if PID_FILE.exists():
        try:
            os.kill(int(PID_FILE.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Recorded server is already running; use status")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "output" / "server.log").open("ab") as log:
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/lightrag-server")],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(str(process.pid))
    print(f"Started PID {process.pid}; log: output/server.log")


def status() -> None:
    with httpx.Client(base_url=base_url(), timeout=10) as client:
        response = client.get("/health")
        response.raise_for_status()
        data = response.json()
    result = {
        key: data.get(key)
        for key in ("status", "working_directory", "input_directory", "webui_available")
    }
    result["workspace"] = data.get("configuration", {}).get("workspace")
    print(json.dumps(result, ensure_ascii=False))


def stop() -> None:
    import psutil

    pid = int(PID_FILE.read_text())
    process = psutil.Process(pid)
    if Path(process.cwd()).resolve() != ROOT or not any(
        "lightrag-server" in item for item in process.cmdline()
    ):
        raise RuntimeError("Recorded PID is not this repository's LightRAG server")
    os.kill(pid, signal.SIGTERM)
    print(f"Requested graceful stop for PID {pid}")


def import_batch(apply: bool) -> None:
    files = checked_files()
    print(
        json.dumps(
            {
                "target": base_url(),
                "workspace": settings()["WORKSPACE"],
                "documents": [
                    {"product": entry["product"], "file": path.name}
                    for path, entry in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not apply:
        return
    if RECEIPTS.exists():
        raise RuntimeError(
            "An upload receipt already exists; inspect original results instead of replaying"
        )
    receipts: list[dict] = []
    with httpx.Client(base_url=base_url(), timeout=120) as client:
        health = client.get("/health")
        health.raise_for_status()
        data = health.json()
        config = settings()
        if (
            data.get("core_version") != "1.5.7"
            or data.get("configuration", {}).get("workspace") != config["WORKSPACE"]
            or Path(data["working_directory"]).resolve() != (ROOT / "rag_storage")
            or Path(data["input_directory"]).resolve() != (ROOT / "inputs")
        ):
            raise RuntimeError(
                "Server version or target does not match this independent batch"
            )
        for path, entry in files:
            receipt = {
                "file": path.name,
                "sha256": entry["sha256"],
                "state": "sending",
                "started_time": time.time(),
            }
            receipts.append(receipt)
            save_json(RECEIPTS, receipts)
            with path.open("rb") as document:
                response = client.post(
                    "/documents/upload", files={"file": (path.name, document)}
                )
            response.raise_for_status()
            receipt.update(state="accepted", response=response.json())
            save_json(RECEIPTS, receipts)
            print(f"Accepted: {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["start", "status", "stop", "preview", "import"]
    )
    args = parser.parse_args()
    actions = {
        "start": start,
        "status": status,
        "stop": stop,
        "preview": lambda: import_batch(False),
        "import": lambda: import_batch(True),
    }
    actions[args.action]()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as exc:
        print(
            f"Operation stopped ({type(exc).__name__}); inspect the local state before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)

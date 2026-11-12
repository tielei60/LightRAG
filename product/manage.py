"""Local process and first-batch operations using the unmodified official API."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "first3"
PID_FILE = OUTPUT / "server.pid"
MANIFEST = OUTPUT / "manifest.json"
RECEIPTS = OUTPUT / "uploads.json"
CONFIG_FILE = ROOT / ".env"


def settings() -> dict[str, str | None]:
    if not CONFIG_FILE.is_file():
        raise ValueError("Selected batch configuration is missing")
    return dict(dotenv_values(CONFIG_FILE, interpolate=False))


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
    expected = manifest.get("expected_product_count", 3)
    if (
        type(expected) is not int
        or expected < 1
        or len({entry["product"] for entry in documents}) != expected
    ):
        raise ValueError("Product scope does not match the selected manifest")
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
        if "prepared_path" in entry:
            prepared = (OUTPUT / entry["prepared_path"]).resolve(strict=True)
            if not prepared.is_relative_to(OUTPUT.resolve()):
                raise ValueError("Prepared input escapes the batch directory")
            if (
                hashlib.sha256(prepared.read_bytes()).hexdigest()
                != entry["prepared_sha256"]
            ):
                raise ValueError("Prepared input changed")
            if json.loads(prepared.read_text()).get("file_source") != path.name:
                raise ValueError("Prepared input source does not match the manifest")
        files.append((path, entry))
    return files


def start() -> None:
    base_url()
    if PID_FILE.exists():
        try:
            os.kill(int(PID_FILE.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Recorded server is already running; use status")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = (
        ROOT / "output/server.log" if OUTPUT.name == "first3" else OUTPUT / "server.log"
    )
    environment = os.environ.copy()
    config = settings()
    environment.update({k: v for k, v in config.items() if v is not None})
    grounded = str(config.get("PRODUCT_GROUNDED_QUERY", "false")).lower() == "true"
    command = (
        [str(ROOT / ".venv/bin/python"), "-m", "product.server"]
        if grounded
        else [str(ROOT / ".venv/bin/lightrag-server")]
    )
    from product.workspace_registry import workspace_writer

    reservation = (
        workspace_writer(ROOT, config["WORKSPACE"])
        if config.get("WORKSPACE")
        else nullcontext()
    )
    # Startup and SDK append use the same reservation. Publish the PID before
    # releasing it, so a following append sees the newly running server.
    with reservation:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        PID_FILE.write_text(str(process.pid))
    print(f"Started PID {process.pid}; log: {log_path.relative_to(ROOT)}")


def status() -> None:
    with httpx.Client(base_url=base_url(), timeout=10, trust_env=False) as client:
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
    command = process.cmdline()
    native_server = any(Path(item).name == "lightrag-server" for item in command)
    product_server = any(
        left == "-m" and right == "product.server"
        for left, right in itertools.pairwise(command)
    )
    if Path(process.cwd()).resolve() != ROOT or not (native_server or product_server):
        raise RuntimeError("Recorded PID is not this repository's LightRAG server")
    os.kill(pid, signal.SIGTERM)
    print(f"Requested graceful stop for PID {pid}")


def import_batch(apply: bool) -> None:
    files = checked_files()
    manifest = json.loads(MANIFEST.read_text())
    if "workspace" in manifest and manifest["workspace"] != settings()["WORKSPACE"]:
        raise ValueError("Batch configuration does not match the manifest workspace")
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
    with httpx.Client(base_url=base_url(), timeout=120, trust_env=False) as client:
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
            if "prepared_path" in entry:
                payload = json.loads((OUTPUT / entry["prepared_path"]).read_text())
                if payload.get("file_source") != path.name:
                    raise ValueError(
                        "Prepared input source does not match the manifest"
                    )
                response = client.post("/documents/text", json=payload)
            else:
                with path.open("rb") as document:
                    response = client.post(
                        "/documents/upload", files={"file": (path.name, document)}
                    )
            response.raise_for_status()
            receipt.update(state="accepted", response=response.json())
            save_json(RECEIPTS, receipts)
            print(f"Accepted: {path.name}", flush=True)


def select_batch(batch: str) -> None:
    global OUTPUT, PID_FILE, MANIFEST, RECEIPTS, CONFIG_FILE
    if not batch or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in batch
    ):
        raise ValueError("Invalid batch name")
    OUTPUT = ROOT / "output" / batch
    PID_FILE = OUTPUT / "server.pid"
    MANIFEST = OUTPUT / "manifest.json"
    RECEIPTS = OUTPUT / "uploads.json"
    CONFIG_FILE = ROOT / ".env" if batch == "first3" else OUTPUT / "runtime.env"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="master")
    parser.add_argument(
        "action", choices=["start", "status", "stop", "preview", "import"]
    )
    args = parser.parse_args()
    select_batch(args.batch)
    actions = {
        "start": start,
        "status": status,
        "stop": stop,
        "preview": lambda: import_batch(False),
        "import": lambda: import_batch(True),
    }
    actions[args.action]()


if __name__ == "__main__":
    # Support direct script invocation as well as ``python -m product.manage``.
    sys.path.insert(0, str(ROOT))
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as exc:
        print(
            f"Operation stopped ({type(exc).__name__}); inspect the local state before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)

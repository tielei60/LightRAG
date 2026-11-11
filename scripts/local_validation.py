#!/usr/bin/env python3
"""Prepare and check an isolated, local-only LightRAG validation instance.

No model requests, document uploads, or data deletion are performed by this tool.
Only `start` runs the existing server; `verify` reads loopback health/UI endpoints.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("data/local-validation")
TEMPLATE = Path("examples/local_validation/env.example")
FIXED = {
    "HOST": "127.0.0.1",
    "PORT": "9631",
    "WORKSPACE": "local_validation",
    "WORKING_DIR": "./rag_storage",
    "INPUT_DIR": "./inputs",
    "LOG_DIR": "./logs",
    "TIKTOKEN_CACHE_DIR": "../../.cache/tiktoken",
    "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
    "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
    "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
    "LLM_BINDING": "openai",
    "EMBEDDING_BINDING": "openai",
    "RERANK_BINDING": "null",
    "NATIVE_MD_IMAGE_DOWNLOAD_ENABLED": "false",
}
MODEL_FIELDS = (
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
)
BLOCKED_PREFIXES = (
    "EXTRACT_LLM_",
    "KEYWORD_LLM_",
    "QUERY_LLM_",
    "VLM_LLM_",
    "LANGFUSE_",
)
# Do not inherit API keys, authentication, telemetry, SSL or logging settings.
# Proxy and trust-store settings are retained so the selected model can be reached.
PROCESS_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LANGUAGE",
    "TZ",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "WINDIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
}
PACKAGES = (
    "lightrag-hku",
    "fastapi",
    "uvicorn",
    "openai",
    "networkx",
    "nano-vectordb",
    "tiktoken",
    "python-dotenv",
)


def runtime_dir(root: Path) -> Path:
    """Reject redirects into another checkout or an existing knowledge base."""
    root = root.resolve()
    for relative in (Path("data"), RUNTIME):
        if (root / relative).is_symlink():
            raise ValueError("The validation data directory must not be a symlink")
    return root / RUNTIME


def initialize(root: Path) -> None:
    runtime = runtime_dir(root)
    path = runtime / ".env"
    if path.exists() or path.is_symlink():
        raise ValueError("Local validation .env already exists; it was not changed")
    runtime.mkdir(parents=True, mode=0o700, exist_ok=True)
    # O_EXCL prevents overwriting a concurrently created file or following a link.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write((root / TEMPLATE).read_text(encoding="utf-8"))
    print("Created data/local-validation/.env with blank model credentials.")


def load_config(root: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    path = runtime_dir(root) / ".env"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Run init first; .env must be a regular local file")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Restrict data/local-validation/.env to mode 600")
    parsed = dotenv_values(path, interpolate=False)
    allowed = set(dotenv_values(root / TEMPLATE, interpolate=False))
    if set(parsed) - allowed:
        raise ValueError("The local validation .env contains unsupported settings")
    return {key: value or "" for key, value in parsed.items()}


def model_issues(config: dict[str, str]) -> list[str]:
    issues = []
    for key in MODEL_FIELDS:
        value = config.get(key, "").strip()
        if not value or any(
            word in value.lower()
            for word in ("your_api", "your-key", "replace_me", "changeme")
        ):
            issues.append(f"Configure {key} locally")
    for key in ("LLM_BINDING_HOST", "EMBEDDING_BINDING_HOST"):
        value = config.get(key, "").strip()
        if not value:
            continue
        try:
            url = urlsplit(value)
            host = url.hostname or ""
            _ = url.port
            local = host in {"localhost", "127.0.0.1", "::1"}
            if (
                not host
                or host.endswith((".invalid", ".example"))
                or host in {"example.com", "example.org", "example.net"}
                or url.username
                or url.password
                or url.query
                or url.fragment
                or not (url.scheme == "https" or (url.scheme == "http" and local))
            ):
                raise ValueError
        except ValueError:
            issues.append(
                f"Use a real HTTPS base URL for {key}; loopback HTTP is allowed"
            )
    try:
        if int(config.get("EMBEDDING_DIM", "0")) <= 0:
            raise ValueError
    except ValueError:
        issues.append("EMBEDDING_DIM must match a positive model output dimension")
    return issues


def check(root: Path, config: dict[str, str], environment: dict[str, str]) -> None:
    runtime = runtime_dir(root)
    for key, expected in FIXED.items():
        if config.get(key) != expected:
            raise ValueError(f"Keep the isolated validation setting {key} unchanged")
    for key, directory in (
        ("WORKING_DIR", "rag_storage"),
        ("INPUT_DIR", "inputs"),
        ("LOG_DIR", "logs"),
    ):
        if (runtime / config[key]).resolve() != runtime / directory:
            raise ValueError(
                f"{key} must not redirect outside the validation directory"
            )
    for directory in ("rag_storage", "inputs", "logs"):
        base = runtime / directory
        if base.exists():
            for parent, directories, files in os.walk(base, followlinks=False):
                if any(
                    (Path(parent) / name).is_symlink() for name in directories + files
                ):
                    raise ValueError(
                        "Validation storage, inputs and logs must not contain symlinks"
                    )
    # python-dotenv normally gives inherited environment variables precedence.
    # Refuse ambiguous settings, and never print their values.
    conflicts = [
        key for key in config if key in environment and environment[key] != config[key]
    ]
    conflicts += [key for key in environment if key.startswith(BLOCKED_PREFIXES)]
    conflicts += [
        key
        for key in ("GUNICORN_CMD_ARGS", "PYTHON_DOTENV_DISABLED")
        if environment.get(key)
    ]
    if conflicts:
        raise ValueError(
            "Use a clean terminal; conflicting environment names: "
            + ", ".join(sorted(set(conflicts)))
        )
    for name in PACKAGES:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(
                f"Missing dependency: {name}; run the documented uv sync"
            ) from None
    for name in ("index.html", "workspace.html"):
        if not (root / "lightrag/api/webui" / name).is_file():
            raise ValueError("Build the WebUI first with the documented Bun commands")


def runtime_environment(
    config: dict[str, str], environment: dict[str, str]
) -> dict[str, str]:
    kept = {
        key: value
        for key, value in environment.items()
        if key in PROCESS_ENV_NAMES or key.startswith("LC_")
    }
    return {**kept, **config, "PYTHON_DOTENV_DISABLED": "1"}


def require_models(config: dict[str, str]) -> None:
    issues = model_issues(config)
    if issues:
        raise ValueError("; ".join(issues))


def verify(root: Path) -> None:
    runtime = runtime_dir(root)
    for path in ("/health", "/webui/", "/workspace/", "/documents/status_counts"):
        connection = http.client.HTTPConnection("127.0.0.1", 9631, timeout=10)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise ValueError(f"Local {path} returned HTTP {response.status}")
            if path == "/health":
                data = json.loads(body)
                if (
                    data.get("status") != "healthy"
                    or data.get("configuration", {}).get("workspace")
                    != "local_validation"
                    or Path(data.get("working_directory", "")).resolve()
                    != runtime / "rag_storage"
                    or Path(data.get("input_directory", "")).resolve()
                    != runtime / "inputs"
                ):
                    raise ValueError(
                        "Port 9631 does not match this isolated validation instance"
                    )
            print(f"PASS {path}")
        finally:
            connection.close()
    print(
        "Local health and UI passed. Real ingestion/query still require the manual sample check."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check", "start", "verify"))
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Reject blank or placeholder model settings during check",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize(ROOT)
            return 0
        config = load_config(ROOT)
        check(ROOT, config, dict(os.environ))
        if args.command == "check":
            if args.require_models:
                require_models(config)
            print(
                "PASS isolated paths, dependency metadata and WebUI files; no model request made."
            )
            if model_issues(config):
                print(
                    "Model configuration is incomplete; fill it only on your local computer."
                )
        elif args.command == "verify":
            verify(ROOT)
        else:
            require_models(config)
            runtime = runtime_dir(ROOT)
            for name in ("inputs", "rag_storage", "logs"):
                (runtime / name).mkdir(exist_ok=True)
            executable = ROOT / ".venv/bin/lightrag-server"
            if not executable.is_file():
                raise ValueError("Missing .venv/bin/lightrag-server")
            # Explicit local settings also prevent imports from discovering a different .env.
            environment = runtime_environment(config, dict(os.environ))
            os.chdir(runtime)
            os.execve(str(executable), [str(executable)], environment)
        return 0
    except (ValueError, OSError, ImportError, json.JSONDecodeError) as error:
        # Never print provider exception strings, config contents, keys, or HTTP bodies.
        if isinstance(error, ValueError):
            print(f"Check failed: {error}", file=sys.stderr)
        else:
            print(
                f"Check failed ({type(error).__name__}); verify local files, dependencies and server state.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

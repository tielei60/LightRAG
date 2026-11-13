"""Start independent database processes using installed database programs."""

import asyncio
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from .common import save_json
from .config import Settings

PG_BIN = Path("/usr/local/opt/postgresql@17/bin")
NEO4J_HOME = Path("/usr/local/opt/neo4j/libexec")


def listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def _run(
    settings: Settings, args: list[str], *, env: dict | None = None, allow_failure=False
):
    result = subprocess.run(
        args,
        check=False,
        env=env,
        cwd=settings.root,
        capture_output=True,
        text=True,
        start_new_session=True,
    )
    text = result.stdout + result.stderr
    for key in ["POSTGRES_PASSWORD", "NEO4J_PASSWORD", "MODEL_API_KEY"]:
        secret = settings.get(key)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    log = settings.root / ".runtime/storage-commands.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(text + "\n")
    if result.returncode and not allow_failure:
        raise RuntimeError(f"B database command failed; see {log}")
    return result.returncode, text


def neo4j_environment(settings: Settings) -> dict[str, str]:
    return {
        **os.environ,
        "NEO4J_HOME": str(NEO4J_HOME),
        "NEO4J_CONF": str(settings.root / ".runtime/neo4j/conf"),
    }


def _neo4j_config(settings: Settings) -> Path:
    root = settings.root / ".runtime/neo4j"
    for name in ["conf", "data", "logs", "run", "import", "plugins"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    values = {
        "server.default_listen_address": "127.0.0.1",
        "server.default_advertised_address": "127.0.0.1",
        "server.bolt.listen_address": "127.0.0.1:17687",
        "server.bolt.advertised_address": "127.0.0.1:17687",
        "server.http.listen_address": "127.0.0.1:17474",
        "server.http.advertised_address": "127.0.0.1:17474",
        "server.https.enabled": "false",
        "server.memory.heap.initial_size": "256m",
        "server.memory.heap.max_size": "768m",
        "server.memory.pagecache.size": "256m",
        "dbms.security.auth_enabled": "true",
        **{
            f"server.directories.{name}": str(root / name)
            for name in ["data", "logs", "run", "import", "plugins"]
        },
    }
    path = root / "conf/neo4j.conf"
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    return path


async def _database(settings: Settings) -> None:
    import asyncpg

    arguments = {
        "host": settings.get("POSTGRES_HOST"),
        "port": settings.integer("POSTGRES_PORT"),
        "user": settings.get("POSTGRES_USER"),
        "password": settings.get("POSTGRES_PASSWORD"),
    }
    connection = await asyncpg.connect(**arguments, database="postgres")
    try:
        if not await connection.fetchval(
            "SELECT 1 FROM pg_database WHERE datname=$1", "insurance_b"
        ):
            await connection.execute('CREATE DATABASE "insurance_b"')
    finally:
        await connection.close()
    connection = await asyncpg.connect(**arguments, database="insurance_b")
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await connection.close()


def postgres_tcp_only(settings: Settings) -> None:
    """B uses loopback TCP; avoid Unix socket length limits in nested app paths."""
    path = settings.root / ".runtime/postgres/postgresql.conf"
    current = path.read_text()
    lines = [
        line
        for line in current.splitlines()
        if not re.match(r"\s*unix_socket_directories\s*=", line)
    ]
    updated = "\n".join(lines) + "\nunix_socket_directories=''\n"
    if updated != current:
        path.write_text(updated)


async def start(settings: Settings) -> dict:
    """Never call global service managers or inspect another instance's data."""
    runtime = settings.root / ".runtime"
    pg = runtime / "postgres"
    runtime.mkdir(parents=True, exist_ok=True)
    if listening(15433) and not (pg / "postmaster.pid").is_file():
        raise RuntimeError(
            "Port 15433 is occupied by a process not owned by this B project"
        )
    neo_pid = runtime / "neo4j/run/neo4j.pid"
    if (listening(17687) or listening(17474)) and not neo_pid.is_file():
        raise RuntimeError("The dedicated B Neo4j ports are occupied")
    if not settings.get("POSTGRES_PASSWORD") or not settings.get("NEO4J_PASSWORD"):
        raise ValueError("B database passwords must be set in config/b.env")
    if not (pg / "PG_VERSION").exists():
        password_file = runtime / "postgres-initial-password"
        password_file.write_text(settings.get("POSTGRES_PASSWORD") + "\n")
        password_file.chmod(0o600)
        try:
            _run(
                settings,
                [
                    str(PG_BIN / "initdb"),
                    "-D",
                    str(pg),
                    "-U",
                    settings.get("POSTGRES_USER"),
                    "--pwfile",
                    str(password_file),
                    "--auth-host=scram-sha-256",
                    "--auth-local=scram-sha-256",
                    "--encoding=UTF8",
                    "--locale=C",
                ],
            )
        finally:
            password_file.unlink(missing_ok=True)
        with (pg / "postgresql.conf").open("a") as stream:
            stream.write(
                "\nport=15433\nlisten_addresses='127.0.0.1'\nshared_buffers='128MB'\nmax_connections=40\n"
            )
    postgres_tcp_only(settings)
    if not listening(15433):
        _run(
            settings,
            [
                str(PG_BIN / "pg_ctl"),
                "-D",
                str(pg),
                "-l",
                str(runtime / "postgres.log"),
                "-w",
                "start",
            ],
        )
    await _database(settings)
    _neo4j_config(settings)
    neo_env = neo4j_environment(settings)
    if not listening(17687):
        auth = runtime / "neo4j/data/dbms/auth.ini"
        if not auth.exists():
            _run(
                settings,
                [
                    str(NEO4J_HOME / "bin/neo4j-admin"),
                    "dbms",
                    "set-initial-password",
                    settings.get("NEO4J_PASSWORD"),
                    "--require-password-change=false",
                ],
                env=neo_env,
            )
        _run(settings, [str(NEO4J_HOME / "bin/neo4j"), "start"], env=neo_env)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and not listening(17687):
        await asyncio.sleep(0.5)
    if not listening(17687):
        raise RuntimeError("B Neo4j has not become ready; inspect .runtime/neo4j/logs")
    report = {
        "postgres_port": 15433,
        "postgres_database": "insurance_b",
        "neo4j_bolt_port": 17687,
        "neo4j_http_port": 17474,
        "postgres_data": str(pg),
        "neo4j_data": str(runtime / "neo4j/data"),
        "independent_instances": True,
    }
    save_json(runtime / "storage.json", report)
    return report


def stop(settings: Settings) -> None:
    pg = settings.root / ".runtime/postgres"
    if (pg / "postmaster.pid").exists():
        _run(
            settings,
            [str(PG_BIN / "pg_ctl"), "-D", str(pg), "-w", "stop", "-m", "fast"],
        )
    if (settings.root / ".runtime/neo4j/run/neo4j.pid").exists():
        _run(
            settings,
            [str(NEO4J_HOME / "bin/neo4j"), "stop"],
            env=neo4j_environment(settings),
        )

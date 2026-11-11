"""Offline safety checks for the local validation preparation helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/local_validation.py"
spec = importlib.util.spec_from_file_location("local_validation", SCRIPT)
assert spec and spec.loader
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    template = tmp_path / local.TEMPLATE
    template.parent.mkdir(parents=True)
    template.write_text((local.ROOT / local.TEMPLATE).read_text())
    webui = tmp_path / "lightrag/api/webui"
    webui.mkdir(parents=True)
    for name in ("index.html", "workspace.html"):
        (webui / name).write_text("<html></html>")
    monkeypatch.setattr(local.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(local, "ROOT", tmp_path)
    local.initialize(tmp_path)
    return tmp_path


def configured(checkout):
    config = local.load_config(checkout)
    for key in local.MODEL_FIELDS:
        config[key] = "test-only-value"
    config["LLM_BINDING_HOST"] = "https://llm.test/v1"
    config["EMBEDDING_BINDING_HOST"] = "https://embedding.test/v1"
    return config


def test_init_is_private_and_never_overwrites(checkout):
    path = checkout / local.RUNTIME / ".env"
    original = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="not changed"):
        local.initialize(checkout)
    assert path.read_bytes() == original


def test_init_rejects_redirected_data(tmp_path):
    outside = tmp_path / "existing_knowledge_base"
    outside.mkdir()
    (tmp_path / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local.initialize(tmp_path)
    assert not list(outside.iterdir())


def test_template_is_safe_but_not_model_ready(checkout):
    config = local.load_config(checkout)
    local.check(checkout, config, {})
    assert all(config[key] == "" for key in local.MODEL_FIELDS)
    with pytest.raises(ValueError, match="Configure LLM_BINDING_HOST locally"):
        local.require_models(config)


def test_load_refuses_exposed_config_permissions(checkout):
    (checkout / local.RUNTIME / ".env").chmod(0o644)
    with pytest.raises(ValueError, match="mode 600"):
        local.load_config(checkout)


def test_load_refuses_unsupported_settings(checkout):
    path = checkout / local.RUNTIME / ".env"
    with path.open("a") as handle:
        handle.write("LANGFUSE_SECRET_KEY=test-only-value\n")
    with pytest.raises(ValueError, match="unsupported settings"):
        local.load_config(checkout)


@pytest.mark.parametrize(
    "key,value",
    [
        ("HOST", "0.0.0.0"),
        ("PORT", "9621"),
        ("WORKING_DIR", "../../old"),
        ("WORKSPACE", "products_first3"),
    ],
)
def test_check_refuses_changes_to_isolation(checkout, key, value):
    config = local.load_config(checkout)
    config[key] = value
    with pytest.raises(ValueError, match=key):
        local.check(checkout, config, {})


def test_check_refuses_storage_symlink(checkout):
    outside = checkout / "original_database"
    outside.mkdir()
    (checkout / local.RUNTIME / "rag_storage").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError, match="redirect outside"):
        local.check(checkout, local.load_config(checkout), {})


@pytest.mark.parametrize(
    "name",
    [
        "LLM_BINDING_API_KEY",
        "WORKING_DIR",
        "EXTRACT_LLM_BINDING_HOST",
        "LANGFUSE_SECRET_KEY",
        "GUNICORN_CMD_ARGS",
    ],
)
def test_environment_conflicts_never_print_values(checkout, name):
    with pytest.raises(ValueError) as result:
        local.check(
            checkout, local.load_config(checkout), {name: "private-test-sentinel"}
        )
    assert name in str(result.value)
    assert "private-test-sentinel" not in str(result.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "https://not-configured.invalid/v1",
        "http://remote.test/v1",
        "https://user:private-test-sentinel@remote.test/v1",
        "https://remote.test/v1?api_key=private-test-sentinel",
        "https://remote.test:bad/v1",
    ],
)
def test_model_url_validation_does_not_expose_credentials(checkout, url):
    config = configured(checkout)
    config["LLM_BINDING_HOST"] = url
    issues = local.model_issues(config)
    assert issues
    assert "private-test-sentinel" not in " ".join(issues)
    assert url not in " ".join(issues)


def test_loopback_model_gateway_is_supported(checkout):
    config = configured(checkout)
    config["LLM_BINDING_HOST"] = "http://127.0.0.1:11434/v1"
    assert local.model_issues(config) == []


def test_embedding_dimension_must_be_positive(checkout):
    config = configured(checkout)
    config["EMBEDDING_DIM"] = "0"
    assert any("EMBEDDING_DIM" in item for item in local.model_issues(config))


def test_check_requires_both_built_ui_entries(checkout):
    (checkout / "lightrag/api/webui/workspace.html").unlink()
    with pytest.raises(ValueError, match="Build the WebUI"):
        local.check(checkout, local.load_config(checkout), {})


def test_check_requires_dependencies(checkout, monkeypatch):
    def missing(name):
        raise local.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(local.importlib.metadata, "version", missing)
    with pytest.raises(ValueError, match="Missing dependency"):
        local.check(checkout, local.load_config(checkout), {})


@pytest.mark.parametrize("wrong_target", [False, True])
def test_verify_checks_target_before_reading_other_routes(
    checkout, monkeypatch, wrong_target
):
    paths = []
    runtime = checkout / local.RUNTIME
    health = {
        "status": "healthy",
        "configuration": {
            "workspace": "original" if wrong_target else "local_validation"
        },
        "working_directory": str(runtime / "rag_storage"),
        "input_directory": str(runtime / "inputs"),
    }

    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", 9631)

        def request(self, method, path):
            assert method == "GET"
            paths.append(path)

        def getresponse(self):
            return self

        status = 200

        def read(self):
            return json.dumps(health).encode()

        def close(self):
            pass

    monkeypatch.setattr(local.http.client, "HTTPConnection", Connection)
    if wrong_target:
        with pytest.raises(ValueError, match="does not match"):
            local.verify(checkout)
        assert paths == ["/health"]
    else:
        local.verify(checkout)
        assert paths == [
            "/health",
            "/webui/",
            "/workspace/",
            "/documents/status_counts",
        ]


def test_start_refuses_blank_credentials_before_exec(checkout, monkeypatch, capsys):
    monkeypatch.setattr(local, "check", lambda *args: None)

    def unexpected(*args):
        pytest.fail("The server must not start with missing model settings")

    monkeypatch.setattr(local.os, "execve", unexpected)
    assert local.main(["start"]) == 1
    assert "Configure LLM_BINDING_API_KEY locally" in capsys.readouterr().err


def test_start_uses_only_isolated_runtime(checkout, monkeypatch):
    config = configured(checkout)
    executable = checkout / ".venv/bin/lightrag-server"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(local, "load_config", lambda root: config)
    monkeypatch.setattr(local, "check", lambda *args: None)
    calls = []
    monkeypatch.setattr(local.os, "chdir", lambda path: calls.append(("cwd", path)))
    monkeypatch.setattr(
        local.os,
        "execve",
        lambda exe, argv, env: calls.append(("exec", exe, argv, env)),
    )
    assert local.main(["start"]) == 0
    assert calls[0] == ("cwd", checkout / local.RUNTIME)
    assert calls[1][1] == str(executable)
    assert calls[1][3]["PYTHON_DOTENV_DISABLED"] == "1"
    assert calls[1][3]["WORKSPACE"] == "local_validation"


@pytest.mark.parametrize(
    "location",
    ["rag_storage/local_validation", "logs/lightrag.log", "inputs/nested/link"],
)
def test_nested_symlinks_cannot_touch_existing_data(checkout, location):
    outside = checkout / "old-data"
    outside.mkdir()
    original = outside / "important.txt"
    original.write_text("original knowledge")
    link = checkout / local.RUNTIME / location
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside if location.endswith("local_validation") else original)
    with pytest.raises(ValueError, match="symlinks"):
        local.check(checkout, local.load_config(checkout), {})
    assert original.read_text() == "original knowledge"


def test_dangling_nested_symlink_is_rejected(checkout):
    link = checkout / local.RUNTIME / "inputs/missing"
    link.parent.mkdir()
    link.symlink_to(checkout / "does-not-exist")
    with pytest.raises(ValueError, match="symlinks"):
        local.check(checkout, local.load_config(checkout), {})


def test_full_start_does_not_inherit_application_settings(checkout, monkeypatch):
    config = configured(checkout)
    executable = checkout / ".venv/bin/lightrag-server"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(local, "load_config", lambda root: config)
    monkeypatch.setattr(
        local.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "LANG": "en_US.UTF-8",
            "HTTPS_PROXY": "http://proxy.test:8080",
            "SSL_CERT_FILE": "/test/ca.pem",
            "AUTH_ACCOUNTS": "test-only-value",
            "LIGHTRAG_API_KEY": "test-only-value",
            "TOKEN_SECRET": "test-only-value",
            "SSL": "true",
            "LOG_LEVEL": "DEBUG",
            "OPENAI_API_KEY": "test-only-value",
            "UI_TEMPLATES_DIR": "/old/ui",
        },
    )
    captured = []
    monkeypatch.setattr(local.os, "chdir", lambda path: None)
    monkeypatch.setattr(local.os, "execve", lambda exe, argv, env: captured.append(env))
    assert local.main(["start"]) == 0
    child = captured[0]
    for key in (
        "AUTH_ACCOUNTS",
        "LIGHTRAG_API_KEY",
        "TOKEN_SECRET",
        "SSL",
        "LOG_LEVEL",
        "OPENAI_API_KEY",
        "UI_TEMPLATES_DIR",
    ):
        assert key not in child
    assert child["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert child["SSL_CERT_FILE"] == "/test/ca.pem"
    assert child["WORKSPACE"] == "local_validation"
    assert child["PYTHON_DOTENV_DISABLED"] == "1"

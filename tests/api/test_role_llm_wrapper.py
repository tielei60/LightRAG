"""Raw role callbacks remain wrapped after a native runtime provider rebuild."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter

pytestmark = pytest.mark.offline


@pytest.fixture
def native_factory(monkeypatch, tmp_path):
    from lightrag.api import config
    from lightrag.llm import openai

    saved_args, saved_initialized = config._global_args, config._initialized
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_MODEL", "base-fixture")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://provider.invalid/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "fixture-only")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    for role in ("EXTRACT", "KEYWORD", "QUERY"):
        monkeypatch.setenv(f"{role}_LLM_BINDING", "openai")
        monkeypatch.setenv(f"{role}_LLM_MODEL", f"{role.lower()}-fixture")
        monkeypatch.setenv(f"{role}_OPENAI_LLM_EXTRA_BODY", '{"enable_thinking":false}')
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    args = config.parse_args()
    args.working_dir = str(tmp_path / "storage")
    args.input_dir = str(tmp_path / "inputs")
    config.initialize_config(args, force=True)
    from lightrag.api import lightrag_server as server

    class FakeRAG:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.roles = kwargs["role_llm_configs"]
            self.builder = None

        def register_role_llm_builder(self, builder):
            self.builder = builder

        def get_llm_role_config(self):
            return {}

    instances = []

    def construct(**kwargs):
        rag = FakeRAG(**kwargs)
        instances.append(rag)
        return rag

    rag_factory = MagicMock(side_effect=construct)
    monkeypatch.setattr(server, "LightRAG", rag_factory)
    monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
    monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
    for route in (
        "create_document_routes",
        "create_query_routes",
        "create_graph_routes",
    ):
        monkeypatch.setattr(server, route, lambda *_a, **_k: APIRouter())
    monkeypatch.setattr(
        server, "OllamaAPI", lambda *_a, **_k: SimpleNamespace(router=APIRouter())
    )
    provider = AsyncMock(return_value="provider answer")
    monkeypatch.setattr(openai, "openai_complete_if_cache", provider)
    try:
        yield server, args, rag_factory, provider, instances
    finally:
        config._global_args, config._initialized = saved_args, saved_initialized


@pytest.mark.asyncio
async def test_default_factory_preserves_native_provider_options(native_factory):
    server, args, rag_factory, provider, _instances = native_factory
    server.create_app(args)
    rag = rag_factory.call_args
    callback = rag.kwargs["role_llm_configs"]["query"].func
    result = await callback("question", system_prompt="sources", stream=False)
    assert result == "provider answer"
    assert provider.await_args.args == ("query-fixture", "question")
    assert provider.await_args.kwargs["system_prompt"] == "sources"
    assert provider.await_args.kwargs["extra_body"] == {"enable_thinking": False}
    assert provider.await_args.kwargs["timeout"] == args.llm_timeout


@pytest.mark.asyncio
async def test_initial_and_runtime_callbacks_are_wrapped_once_before_queue(
    native_factory,
):
    server, args, rag_factory, provider, instances = native_factory
    created = []

    def wrapper(role, raw):
        created.append((role, raw))
        if role != "query":
            return raw

        async def query(*a, **kw):
            return "wrapped:" + await raw(*a, **kw)

        query.__wrapped__ = raw
        return query

    server.create_app(args, role_llm_func_wrapper=wrapper)
    configs = rag_factory.call_args.kwargs["role_llm_configs"]
    assert [role for role, _ in created] == [spec.name for spec in server.ROLES]
    assert configs["query"].max_async == args.query_llm_max_async
    initial = configs["query"].func
    assert await initial("first", stream=False) == "wrapped:provider answer"
    assert configs["extract"].func is created[0][1]
    # The registered builder closes over the hook; it rebuilds raw provider
    # closures rather than wrapping the previous queue-managed callback.
    rebuilt, kwargs = instances[0].builder(
        "query",
        {
            "model": "replacement-model",
            "timeout": 17,
            "provider_options": {"extra_body": {"enable_thinking": False}},
        },
    )
    assert kwargs == {}
    assert len(created) == len(server.ROLES) + 1
    assert created[-1][0] == "query"
    assert rebuilt.__wrapped__ is created[-1][1]
    assert rebuilt.__wrapped__ is not initial
    assert await rebuilt("second", stream=False) == "wrapped:provider answer"
    assert provider.await_args.args[0] == "replacement-model"
    assert provider.await_args.kwargs["timeout"] == 17
    assert provider.await_count == 2


@pytest.mark.parametrize("wrapper", [False, lambda _role, _raw: None])
def test_wrapper_must_be_callable_and_return_callable(native_factory, wrapper):
    server, args, rag_factory, _provider, _instances = native_factory
    with pytest.raises(TypeError, match="callable"):
        server.create_app(args, role_llm_func_wrapper=wrapper)
    assert rag_factory.call_count == 0


def test_native_launcher_forwards_wrapper_without_replacing_lifecycle(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api import config
    from lightrag.api import lightrag_server as server

    monkeypatch.delenv("GUNICORN_CMD_ARGS", raising=False)
    monkeypatch.setattr(config, "initialize_config", lambda: None)
    args = SimpleNamespace(host="127.0.0.1", port=9621, ssl=False)
    monkeypatch.setattr(server, "global_args", args)
    monkeypatch.setattr(server, "check_env_file", lambda: True)
    for name in (
        "check_and_install_dependencies",
        "configure_logging",
        "update_uvicorn_mode_config",
    ):
        monkeypatch.setattr(server, name, lambda: None)
    monkeypatch.setattr(server, "display_splash_screen", lambda *_args: None)
    native_app = object()
    factory = MagicMock(return_value=native_app)
    run = MagicMock()
    monkeypatch.setattr(server, "create_app", factory)
    monkeypatch.setattr(server.uvicorn, "run", run)
    wrapper = lambda _role, raw: raw
    server.main(role_llm_func_wrapper=wrapper)
    factory.assert_called_once_with(args, role_llm_func_wrapper=wrapper)
    assert run.call_args.kwargs["app"] is native_app
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert "root_path" not in run.call_args.kwargs

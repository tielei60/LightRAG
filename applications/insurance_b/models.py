"""Explicit providers and native SDK construction for the B instances."""

import time
import weakref
from pathlib import Path

from .common import append_event, digest, read_json
from .config import PROJECT_ROOT, ROOT, Settings
from .contracts import contract_module


def bind_extraction_cache_scope(rag, extraction):
    """Partition native reads/writes by the evidence adapter's document context."""
    native_builder = type(rag)._build_global_config
    reference = weakref.ref(rag)

    def build_config():
        current = reference()
        if current is None:
            raise RuntimeError("The B extraction owner was released")
        config = native_builder(current)
        config["llm_cache_identities"]["extract"] = {
            **config["llm_cache_identities"]["extract"],
            "b_evidence_source": extraction.cache_identity(),
        }
        return config

    rag._build_global_config = build_config


def provider(settings: Settings, role: str):
    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        settings.require_models()
        from lightrag.llm.openai import openai_complete_if_cache

        kwargs.setdefault("extra_body", {"enable_thinking": False})
        kwargs.setdefault("temperature", float(settings.get("LLM_TEMPERATURE", "0")))
        if role == "extract":
            kwargs.setdefault("stream", False)
            kwargs.setdefault("response_format", {"type": "json_object"})
        started = time.monotonic()

        class Usage:
            def __init__(self):
                self.rows = []

            def add_usage(self, usage):
                self.rows.append(usage)

        usage = Usage()
        kwargs.setdefault("token_tracker", usage)
        append_event(
            settings.run_dir / "provider-events.jsonl",
            {
                "role": role,
                "state": "started",
                "model": settings.get(role.upper() + "_MODEL"),
                "stream": bool(kwargs.get("stream")),
                "prompt_sha256": digest(prompt),
            },
        )
        response = await openai_complete_if_cache(
            settings.get(role.upper() + "_MODEL"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=settings.get("OPENAI_BASE_URL"),
            api_key=settings.model_key(role),
            **kwargs,
        )
        append_event(
            settings.run_dir / "provider-events.jsonl",
            {
                "role": role,
                "state": "response_received",
                "stream": hasattr(response, "__aiter__"),
                "elapsed_seconds": time.monotonic() - started,
                "usage": usage.rows,
            },
        )
        if hasattr(response, "__aiter__"):

            async def observed_stream():
                try:
                    async for token in response:
                        yield token
                finally:
                    append_event(
                        settings.run_dir / "provider-events.jsonl",
                        {
                            "role": role,
                            "state": "stream_consumed",
                            "elapsed_seconds": time.monotonic() - started,
                            "usage": usage.rows,
                        },
                    )

            return observed_stream()
        return response

    return complete


def make_rag(
    settings: Settings, *, extraction=None, audited_query=True, query_record=None
):
    settings.activate()
    import lightrag

    if not Path(lightrag.__file__).resolve().is_relative_to(PROJECT_ROOT / "lightrag"):
        raise ValueError("B uses the shared SDK from this project")
    frozen = read_json(settings.contract_snapshot)
    actual_sdk = {
        str(path.relative_to(PROJECT_ROOT)): digest(path.read_bytes())
        for path in sorted((PROJECT_ROOT / "lightrag").rglob("*.py"))
    }
    if actual_sdk != frozen["active_sdk"]["source_sha256"]:
        raise ValueError(
            "The shared SDK changed since the B snapshot; freeze a new B run explicitly"
        )
    package = (
        "_current" if settings.get("CONTRACT_REVISION") == "current" else "_frozen"
    )
    for name, expected in frozen["contract_modules"].items():
        if (
            digest((ROOT / package / (name + ".py")).read_bytes())
            != expected["frozen_sha256"]
        ):
            raise ValueError(
                "A frozen evidence or answer contract changed; freeze a new variant explicitly"
            )
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_embed
    from lightrag.llm_roles import RoleLLMConfig
    from lightrag.rerank import ali_rerank
    from lightrag.utils import EmbeddingFunc

    async def embed(texts, embedding_dim=None, max_token_size=None, **kwargs):
        settings.require_models()
        return await openai_embed.func(
            texts,
            model=settings.get("EMBEDDING_MODEL"),
            base_url=settings.get("EMBEDDING_BASE_URL")
            or settings.get("OPENAI_BASE_URL"),
            api_key=settings.model_key("EMBEDDING"),
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            **kwargs,
        )

    async def rerank(query, documents, top_n=None, **kwargs):
        settings.require_models()
        return await ali_rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=settings.get("RERANK_MODEL"),
            api_key=settings.model_key("RERANK"),
            base_url=settings.get("RERANK_URL"),
            **kwargs,
        )

    query = provider(settings, "query")
    if audited_query:
        AnswerQuery = contract_module(settings, "answer_query").AnswerQuery
        provider_limits = contract_module(settings, "grounded_query").provider_limits
        record_query = query_record or (
            lambda event: append_event(settings.run_dir / "answer-events.jsonl", event)
        )
        if settings.get("ANSWER_GUARD") == "scope-v1":
            from .query_guard import guarded_provider

            query = guarded_provider(settings, query, record_query)

        query = AnswerQuery(
            query,
            record=record_query,
            answer_instructions=(
                settings.profiles / "user_prompt/products.md"
            ).read_text(),
            **provider_limits(
                {"QUERY_MAX_SECONDS": settings.get("QUERY_MAX_SECONDS", "180")}
            ),
        )
    funcs = {
        "extract": extraction or provider(settings, "extract"),
        "keyword": provider(settings, "keyword"),
        "query": query,
    }
    roles = {
        role: RoleLLMConfig(
            func=func,
            max_async=4 if role == "extract" else 2,
            timeout=settings.integer("LLM_TIMEOUT"),
            metadata={
                "binding": "openai",
                "model": settings.get(role.upper() + "_MODEL"),
                "provider_options": {},
            },
        )
        for role, func in funcs.items()
    }
    rag = LightRAG(
        working_dir=str(settings.root / "rag_storage"),
        workspace=settings.workspace,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
        doc_status_storage="PGDocStatusStorage",
        llm_model_func=provider(settings, "extract"),
        role_llm_configs=roles,
        embedding_func=EmbeddingFunc(
            embedding_dim=settings.integer("EMBEDDING_DIM"),
            max_token_size=int(settings.get("EMBEDDING_TOKEN_LIMIT", "8192")),
            send_dimensions=settings.get("EMBEDDING_SEND_DIM", "true").lower()
            == "true",
            func=embed,
            model_name=settings.get("EMBEDDING_MODEL"),
        ),
        rerank_model_func=rerank,
        entity_extraction_use_json=True,
        entity_extract_max_gleaning=0,
        entity_extract_max_records=settings.integer("MAX_EXTRACTION_RECORDS"),
        entity_extract_max_entities=settings.integer("MAX_EXTRACTION_ENTITIES"),
        addon_params={"language": "Chinese", "entity_type_prompt_file": "products.yml"},
        summary_max_tokens=1200,
        summary_context_size=8192,
        force_llm_summary_on_merge=3,
        chunk_token_size=8192,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        enable_llm_cache=False,
    )
    if extraction is not None:
        extraction.bind(rag)
        bind_extraction_cache_scope(rag, extraction)
        if settings.get("FINAL_CONTEXT") == "source-v1":
            from .input_adapters.source_context import install

            install(settings, extraction)
    if settings.get("STRUCTURE_REPAIR") in {"headers-v1", "headers-math-v2"}:
        from .structure_repair import install_parsers

        install_parsers()
        rag._insurance_b_root = settings.root
        rag._insurance_b_structure_repair = settings.get("STRUCTURE_REPAIR")
    return rag


def query_parameters(settings: Settings, *, stream=False):
    from lightrag import QueryParam

    return QueryParam(
        mode="mix",
        top_k=settings.integer("TOP_K"),
        chunk_top_k=settings.integer("CHUNK_TOP_K"),
        max_entity_tokens=settings.integer("MAX_ENTITY_TOKENS"),
        max_relation_tokens=settings.integer("MAX_RELATION_TOKENS"),
        max_total_tokens=settings.integer("MAX_TOTAL_TOKENS"),
        enable_rerank=True,
        conversation_history=[],
        stream=stream,
    )


async def initialize_rag(rag, settings: Settings):
    await rag.initialize_storages()
    from lightrag.kg.shared_storage import initialize_pipeline_status

    await initialize_pipeline_status(workspace=settings.workspace)

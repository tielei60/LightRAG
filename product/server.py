"""Run the native server with a source-grounded natural answer callback."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightrag.prompt import load_user_prompt_prefix_source
from product.answer_query import AnswerQuery
from product.grounded_query import provider_limits


def record_grounding_event(event: dict[str, Any]) -> None:
    """Keep optional same-request evidence diagnostics in the selected batch."""
    location = os.getenv("PRODUCT_GROUNDING_LOG")
    if not location:
        return
    path = Path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n"
        )


def wrap_query_role(role: str, provider_func: Callable) -> Callable:
    """Keep retrieval/extraction providers native and constrain final answers."""
    if role != "query":
        return provider_func
    return AnswerQuery(
        provider_func,
        record=record_grounding_event,
        answer_instructions=load_user_prompt_prefix_source(),
        **provider_limits(os.environ),
    )


def main() -> None:
    # An answer-cache hit happens before the query callback. Every answer
    # must be generated from the current request's retrieved source context.
    os.environ["ENABLE_LLM_CACHE"] = "false"
    from lightrag.api.lightrag_server import main as native_main

    native_main(role_llm_func_wrapper=wrap_query_role)


if __name__ == "__main__":
    main()

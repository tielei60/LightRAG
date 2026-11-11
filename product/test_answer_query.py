"""Natural model answers, real transport timing, and same-request citation bounds."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from product.answer_query import ANSWER_CONTRACT_VERSION, AnswerError, AnswerQuery
from product.test_grounded_query import native_prompt, wrapped


def streaming_provider(tokens):
    async def generate():
        for token in tokens:
            yield token

    return AsyncMock(side_effect=lambda *args, **kwargs: generate())


def provider_documents(provider):
    system = provider.call_args.kwargs["system_prompt"]
    block = system.split("<source_documents>\n", 1)[1].split(
        "\n</source_documents>", 1
    )[0]
    return json.loads(block)


def source_context():
    return native_prompt(
        [
            {"reference_id": "1", "content": "甲责任只适用事件发生后七日。"},
            {"reference_id": "2", "content": "乙责任只适用事件发生后三十日。"},
        ],
        references={"1": "甲条款.pdf", "2": "乙条款.pdf"},
        graph="错误的图谱概述称所有责任期间完全相同。",
    )


@pytest.mark.asyncio
async def test_single_model_call_authors_direct_answer_instead_of_copying_source_body():
    tokens = ["两项责任的适用期限不同", "，不能直接视为相同。[1][2]"]
    provider = streaming_provider(tokens)
    events = []
    callback = AnswerQuery(provider, record=events.append)
    history = [{"role": "user", "content": "继续比较前述两项责任"}]
    answer = await callback(
        "这两项期限完全相同吗？",
        system_prompt=source_context(),
        history_messages=history,
        stream=False,
        hashing_kv="native-cache",
        response_format={"type": "json_object"},
        enable_cot=True,
    )
    assert answer.startswith("".join(tokens))
    assert "> " not in answer
    assert "甲责任只适用事件发生后七日" not in answer
    assert "乙责任只适用事件发生后三十日" not in answer
    assert answer.endswith("### References\n\n- [1] 甲条款.pdf\n- [2] 乙条款.pdf")
    provider.assert_awaited_once()
    request = provider.call_args
    assert request.kwargs["stream"] is True
    assert request.kwargs["enable_cot"] is False
    assert request.kwargs["history_messages"] == history
    assert request.kwargs["hashing_kv"] == "native-cache"
    assert "response_format" not in request.kwargs
    assert request.args[0] == "这两项期限完全相同吗？"
    documents = provider_documents(provider)
    assert [
        excerpt["text"]
        for document in documents.values()
        for excerpt in document["excerpts"]
    ] == [
        "甲责任只适用事件发生后七日。",
        "乙责任只适用事件发生后三十日。",
    ]
    assert "错误的图谱概述" not in request.args[0]
    assert "错误的图谱概述" not in request.kwargs["system_prompt"]
    assert callback.cache_identity == {"response_adapter": ANSWER_CONTRACT_VERSION}
    assert events[0]["prompt"] == request.args[0]
    assert events[0]["system_prompt"] == request.kwargs["system_prompt"]
    assert events[0]["history_messages"] == history
    assert events[0]["native_system_prompt"] == source_context()
    assert events[0]["source_documents"] == documents
    assert [
        event["chunk"] for event in events if event["state"] == "provider_chunk"
    ] == tokens
    assert events[-1]["state"] == "completed"
    assert events[-1]["provider_answer"] == "".join(tokens)
    assert events[-1]["answer"] == answer
    assert events[-1]["cited_ids"] == ["1", "2"]


@pytest.mark.asyncio
async def test_entire_source_units_keep_navigation_headings_and_meaningful_parents_for_model():
    title = "# 测试产品投保及核保规则"
    parent = "可选责任须先投保基本责任，且年龄未满60周岁。"
    body = "符合约定时按日给付住院津贴。"
    content = wrapped(body) + (
        f"\n【同文件引用原文：structural-parent；Unicode码点10:20】\n{title}\n【引用原文结束】"
        f"\n【同文件引用原文：structural-parent；Unicode码点30:80】\n{parent}\n【引用原文结束】"
    )
    provider = streaming_provider(
        ["可以选择津贴责任，但须满足先选基本责任及年龄条件。[1]"]
    )
    answer = await AnswerQuery(provider)(
        "能否单独选津贴？",
        system_prompt=native_prompt([{"reference_id": "1", "content": content}]),
    )
    documents = provider_documents(provider)
    assert list(documents) == ["[1]"]
    units = documents["[1]"]["excerpts"]
    assert [unit["text"] for unit in units] == [body, title, parent]
    assert units[0]["governing_context"] == [title, parent]
    assert all(set(unit) == {"role", "text", "governing_context"} for unit in units)
    assert title not in answer and body not in answer
    assert "先选基本责任及年龄条件" in answer


@pytest.mark.asyncio
async def test_model_input_has_only_file_citation_namespace_and_complete_governing_text():
    table = "| 计划 | 责任 |\n| --- | --- |\n| 甲 | 责任A |"
    title = "# 原始计划表标题"
    prerequisite = "本项须先投保基本责任，且适用约定年龄范围。"
    definition = "同一事故指约定原因导致的连续事故。"
    content = wrapped(table) + (
        f"\n【同文件引用原文：structural-parent；Unicode码点10:20】\n{title}\n【引用原文结束】"
        f"\n【同文件引用原文：structural-parent；Unicode码点30:70】\n{prerequisite}\n【引用原文结束】"
        f"\n【同文件引用原文：explicit-definition-reference；Unicode码点80:100】\n{definition}\n【引用原文结束】"
    )
    native = native_prompt(
        [
            {"reference_id": "3", "content": content},
            {"reference_id": "1", "content": "另一文件中的完整条款。"},
        ],
        references={"3": "计划表.md", "1": "条款.pdf"},
    )
    provider = streaming_provider(["计划表列有对应责任，选择时仍须符合前提。[3]"])
    events = []
    answer = await AnswerQuery(provider, record=events.append)(
        "计划表说明了什么？", system_prompt=native
    )
    assert provider.call_args.args[0] == "计划表说明了什么？"
    payload_text = provider.call_args.kwargs["system_prompt"]
    documents = provider_documents(provider)
    assert list(documents) == ["[1]", "[3]"]
    assert documents["[3]"]["source_file"] == "计划表.md"
    assert [excerpt["text"] for excerpt in documents["[3]"]["excerpts"]] == [
        table,
        title,
        prerequisite,
        definition,
    ]
    assert documents["[3]"]["excerpts"][0]["governing_context"] == [title, prerequisite]
    assert all(
        set(document) == {"source_file", "product_name", "excerpts"}
        for document in documents.values()
    )
    assert documents["[3]"]["product_name"] == "测试产品"
    assert documents["[1]"]["product_name"] is None
    assert "source_units" not in payload_text
    for internal_key in (
        "U0001",
        "identifier",
        "reference_id",
        "required_unit_ids",
        "chunk_index",
    ):
        assert internal_key not in payload_text
    # The first source unit belongs to file 3, while detailed IDs remain only
    # in diagnostics. The model never sees a competing unit-number namespace.
    assert events[0]["source_units"][0]["identifier"] == "U0001"
    assert events[0]["source_units"][0]["reference_id"] == "3"
    system = provider.call_args.kwargs["system_prompt"]
    assert "本次可用的文件引用标签只有：[1]、[3]。" in system
    assert "reference_id" not in system and "required_unit_ids" not in system
    assert answer.endswith("### References\n\n- [3] 计划表.md")


@pytest.mark.asyncio
async def test_configured_answer_instructions_reach_model_and_exact_prompt_trace():
    instructions = "先直接回答，再用简洁列表说明差异；由问答模型决定呈现方式。"
    events = []
    provider = streaming_provider(["两项规定应分别核查。[1][2]"])
    callback = AnswerQuery(
        provider, record=events.append, answer_instructions=instructions
    )
    await callback("问题", system_prompt=source_context())
    system = provider.call_args.kwargs["system_prompt"]
    assert instructions in system
    assert system.index(instructions) < system.index("<source_documents>")
    assert system.index("</source_documents>") < system.index("输出前简短核对")
    assert "不能编造数值、来源或References列表。" in system
    assert "本次可用的文件引用标签只有：[1]、[2]。" in system
    assert events[0]["system_prompt"] == system


def test_answer_instructions_must_be_resolved_text_not_a_source_object():
    with pytest.raises(TypeError, match="resolved text"):
        AnswerQuery(AsyncMock(), answer_instructions=object())


@pytest.mark.asyncio
async def test_tokens_arrive_before_provider_finishes_without_pacing_or_extra_calls():
    blocked, release = asyncio.Event(), asyncio.Event()

    async def generate():
        yield "两项责任的期限"
        blocked.set()
        await release.wait()
        yield "不同。[1][2]"

    provider = AsyncMock(side_effect=lambda *args, **kwargs: generate())
    events = []
    callback = AnswerQuery(provider, record=events.append)
    stream = await callback("是否相同？", system_prompt=source_context(), stream=True)
    provider.assert_not_awaited()
    first = await anext(stream)
    assert first == "两项责任的期限"
    assert not blocked.is_set()
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(blocked.wait(), 1)
    assert not pending.done()
    assert not any(event["state"] == "completed" for event in events)
    release.set()
    second = await pending
    tail = [part async for part in stream]
    assert second == "不同。[1][2]"
    assert len(tail) == 1 and tail[0].startswith("\n\n### References")
    assert provider.await_count == 1
    assert events[-1]["state"] == "completed"


@pytest.mark.asyncio
async def test_stream_nonstream_preserve_model_text_and_use_only_actually_cited_files():
    provider = streaming_provider(["乙项的规定需单独核查。", "[", "2", "]"])
    callback = AnswerQuery(provider)
    text = await callback("问题", system_prompt=source_context())
    streamed = await callback("问题", system_prompt=source_context(), stream=True)
    assert "".join([part async for part in streamed]) == text
    assert "[2] 乙条款.pdf" in text and "甲条款.pdf" not in text
    # Exact repeats still use the configured model; old extracted answers and
    # earlier natural answers never bypass the current request's generation.
    assert provider.await_count == 2
    assert all(call.kwargs["stream"] is True for call in provider.call_args_list)


@pytest.mark.asyncio
async def test_file_list_alone_cannot_authorize_citation_without_an_eligible_source_unit():
    provider = streaming_provider(["这份文件未提供任何原文。[2]"])
    native = native_prompt(
        [{"reference_id": "1", "content": "有依据的完整原文。"}],
        references={"1": "有原文.pdf", "2": "只出现在文件列表.pdf"},
    )
    stream = await AnswerQuery(provider)("问题", system_prompt=native, stream=True)
    with pytest.raises(AnswerError, match="本次资料之外"):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["参考资料", "资料来源"])
async def test_reference_words_at_token_boundary_remain_valid_ordinary_prose(prefix):
    tokens = [prefix, "显示两项期限不同。[1][2]"]
    provider = streaming_provider(tokens)
    stream = await AnswerQuery(provider)(
        "问题", system_prompt=source_context(), stream=True
    )
    assert await anext(stream) == prefix
    assert await anext(stream) == tokens[1]
    assert "### References" in "".join([part async for part in stream])


@pytest.mark.asyncio
async def test_reference_heading_on_unterminated_final_line_is_rejected_at_eof():
    provider = streaming_provider(["已给出依据。[1]\n参考资料"])
    stream = await AnswerQuery(provider)(
        "问题", system_prompt=source_context(), stream=True
    )
    assert await anext(stream) == "已给出依据。[1]\n参考资料"
    with pytest.raises(AnswerError, match="自行生成的来源列表"):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens",
    [
        [
            "年度合计为[给付的生存保险金 + ",
            "基本保险金额减少部分对应的现金价值]，",
            "须符合原文约定的比例上限。[1]",
        ],
        ["说明[仅用于解释]，分组[2 + 3]不代表引用。[1]"],
        ["公式分组[现金价值 × [约定比例]]应保留。[1]"],
        ["按[2", " + 3]分组说明。[", "1", "]"],
        ["普通全角分组［仅供说明］应保留。[1]"],
        ["算式[1/2]和[2 - 1]按分组保留。[1]"],
    ],
)
async def test_formula_and_ordinary_brackets_preserve_real_stream_and_file_citations(
    tokens,
):
    provider = streaming_provider(tokens)
    events = []
    stream = await AnswerQuery(provider, record=events.append)(
        "减保额度如何约定？", system_prompt=source_context(), stream=True
    )
    for token in tokens:
        assert await anext(stream) == token
    footer = "".join([part async for part in stream])
    assert footer.endswith("### References\n\n- [1] 甲条款.pdf")
    assert events[-1]["state"] == "completed"
    assert events[-1]["provider_answer"] == "".join(tokens)
    assert events[-1]["cited_ids"] == ["1"]
    provider.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_tail",
    [
        "未提供来源。[9]",
        "将摘录序号误作引用。[19]",
        "混合编号。[1,2]",
        "全角编号。［1］",
        "混合括号。[1］",
        "非ASCII数字。[１]",
        "摘录内部编号。[U0019]",
        "公式不能隐藏引用。[现金价值 × [9]]",
        "公式不能隐藏摘录编号。[现金价值 × [U0019]]",
        "其他名称。[外部报告]",
        "\n### References\n- [1] 自造文件.pdf",
        "\n参考文献\n外部资料",
        "外链[1](https://foreign.example)",
    ],
)
async def test_invalid_citation_or_generated_reference_list_marks_partial_answer_incomplete(
    invalid_tail,
):
    provider = streaming_provider(["先核对各自约定。[1]", invalid_tail])
    events = []
    callback = AnswerQuery(provider, record=events.append)
    stream = await callback("问题", system_prompt=source_context(), stream=True)
    assert await anext(stream) == "先核对各自约定。[1]"
    with pytest.raises(AnswerError):
        await anext(stream)
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["provider_answer"].endswith(invalid_tail)
    assert events[-1]["emitted_answer"] == "先核对各自约定。[1]"
    assert not any(event["state"] == "completed" for event in events)
    assert provider.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [""],
        ["没有引用的模型答复"],
        ["只有公式[2 + 3]没有文件引用"],
        ["正文[1]，引用未结束[2\n"],
        ["正文[1]，引用未结束[U", "0019"],
        ["正文[1]，引用未结束[1,"],
        ["正文[1]，引用未结束[U0019,"],
        ["正文[1]，引用未结束[来源："],
    ],
)
async def test_empty_uncited_or_unfinished_answer_is_explicit_failure(tokens):
    provider = streaming_provider(tokens)
    events = []
    stream = await AnswerQuery(provider, record=events.append)(
        "问题", system_prompt=source_context(), stream=True
    )
    parts = []
    with pytest.raises(AnswerError):
        async for part in stream:
            parts.append(part)
    assert all("### References" not in part for part in parts)
    assert events[-1]["state"] == "incomplete"
    assert not any(event["state"] == "completed" for event in events)


@pytest.mark.asyncio
async def test_closing_early_closes_provider_and_never_reports_completed():
    closed = asyncio.Event()

    async def generate():
        try:
            yield "两项责任不能混同。[1]"
            yield "还有不同的适用范围。[2]"
        finally:
            closed.set()

    events = []
    provider = AsyncMock(side_effect=lambda *args, **kwargs: generate())
    callback = AnswerQuery(provider, record=events.append, provider_max_async=1)
    stream = await callback("问题", system_prompt=source_context(), stream=True)
    await anext(stream)
    await stream.aclose()
    assert closed.is_set()
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["error_type"] == "GeneratorExit"
    result = await asyncio.wait_for(callback("问题", system_prompt=source_context()), 1)
    assert "### References" in result and provider.await_count == 2


@pytest.mark.asyncio
async def test_cancelled_read_closes_provider_and_releases_slot_without_success():
    waiting, closed = asyncio.Event(), asyncio.Event()
    first = True

    async def generate():
        nonlocal first
        try:
            yield "可先核查约定。[1]"
            if first:
                first = False
                waiting.set()
                await asyncio.Event().wait()
            yield "另一项单独核查。[2]"
        finally:
            closed.set()

    provider = AsyncMock(side_effect=lambda *args, **kwargs: generate())
    events = []
    callback = AnswerQuery(provider, record=events.append, provider_max_async=1)
    stream = await callback("问题", system_prompt=source_context(), stream=True)
    await anext(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(waiting.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed.is_set()
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["error_type"] == "CancelledError"
    assert not any(event["state"] == "completed" for event in events)
    result = await asyncio.wait_for(callback("问题", system_prompt=source_context()), 1)
    assert "### References" in result and provider.await_count == 2


@pytest.mark.asyncio
async def test_timeout_during_real_stream_is_visible_and_releases_provider_slot():
    closed = asyncio.Event()
    first = True

    async def generate():
        nonlocal first
        try:
            yield "已有依据。[1]"
            if first:
                first = False
                await asyncio.Event().wait()
        finally:
            closed.set()

    provider = AsyncMock(side_effect=lambda *args, **kwargs: generate())
    events = []
    callback = AnswerQuery(
        provider, record=events.append, provider_max_async=1, provider_timeout=0.05
    )
    stream = await callback("问题", system_prompt=source_context(), stream=True)
    assert await anext(stream) == "已有依据。[1]"
    with pytest.raises(AnswerError, match="问答生成超时"):
        await anext(stream)
    assert closed.is_set() and events[-1]["state"] == "incomplete"
    result = await asyncio.wait_for(callback("问题", system_prompt=source_context()), 1)
    assert "### References" in result and provider.await_count == 2


@pytest.mark.asyncio
async def test_limit_is_held_until_active_provider_stream_finishes():
    release, second_attempted = asyncio.Event(), asyncio.Event()
    entered = []

    async def provider(prompt, **kwargs):
        question = prompt
        entered.append(question)

        async def generate():
            yield "依据各自条款回答。[1]"
            if question == "甲":
                await release.wait()

        return generate()

    async def advance(stream):
        second_attempted.set()
        return await anext(stream)

    callback = AnswerQuery(provider, provider_max_async=1)
    first = await callback("甲", system_prompt=source_context(), stream=True)
    second = await callback("乙", system_prompt=source_context(), stream=True)
    await anext(first)
    pending = asyncio.create_task(advance(second))
    await asyncio.wait_for(second_attempted.wait(), 1)
    assert entered == ["甲"] and not pending.done()
    release.set()
    assert "### References" in await anext(first)
    await pending
    assert entered == ["甲", "乙"]
    assert [part async for part in first] == []
    await second.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("provider failed"), TimeoutError(), RuntimeError()]
)
async def test_initial_provider_failure_is_raised_inside_iterator_with_visible_message(
    error,
):
    provider = AsyncMock(side_effect=error)
    events = []
    stream = await AnswerQuery(provider, record=events.append)(
        "问题", system_prompt=source_context(), stream=True
    )
    with pytest.raises(Exception) as caught:
        await anext(stream)
    assert str(caught.value)
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["provider_answer"] == ""
    assert provider.await_count == 1


@pytest.mark.asyncio
async def test_buffered_provider_string_is_rejected_instead_of_fake_streamed():
    provider = AsyncMock(return_value="完整但缓冲的答案。[1]")
    stream = await AnswerQuery(provider)(
        "问题", system_prompt=source_context(), stream=True
    )
    with pytest.raises(AnswerError, match="实时文本流"):
        await anext(stream)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider_max_async": 0},
        {"provider_timeout": 0},
        {"provider_timeout": float("nan")},
    ],
)
def test_invalid_provider_limits_fail_before_request(kwargs):
    with pytest.raises(ValueError, match="positive"):
        AnswerQuery(AsyncMock(), **kwargs)

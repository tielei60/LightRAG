"""Shared experiment guard derived exclusively from same-request source text."""

import json
import re
import unicodedata

import tiktoken

from .common import digest
from .contracts import contract_module

VERSION = "source-scope-v1"
POLICY = """补充来源约束：以下结构索引只定位本次原文，不新增保险事实。
分别核对各计划和各项责任的原文区间；责任免除、产品总体利益和后续章节不属于最后一个计划的积极责任清单。
不得从通常的保险做法推定未列责任；完整计划区间仅列某项保险金时，直接说明原文明示清单，不能借其他计划或产品概述补上身故等责任。
责任之间有关联、依附关系或共用免赔额，不证明保险期间延续规则也共同适用。只在对应责任原文明示时才声明其延续；住院前后就诊窗口与保险期间届满后的延续是两个不同约定，不能互相替代，也不能推断随另一责任自动延续。
原文的超过是严格大于；达到包含等于。问题只达到边界金额时，不得先无条件肯定必须办理，再引用超过门槛作为理由；应明确区分恰好等于和超过，保留其他原文条件。
仅回答用户所问产品及属性。不扩展未问产品的承保/给付结论，不添加无明确依据的所有计划适用声明、责任叠加或收益高低判断。
本次未检索到不能推出全部资料或知识库没有该产品；不要据此要求用户更换产品名称或列一串未问产品。
只引用能在对应责任原文区间核对的条款号，无法核对号码时仅引用文件。不能借相邻章节号定位另一责任。
每次重述公式须保留指数、计算基数及适用阶段；纯增长因子比较不必附带展开三项取大的规则。可能、非保证、超过等限定不可强化或省略。
detected_shared_guard_conflicts非空时须对照原文修正真实矛盾，不得在保留该矛盾时宣称全通过。
审核issues只列具体原文反例，最多8项，每项不超过600字；不得输出自我辩论、反复猜测或认可答案正确的推理过程。找不到具体错误时应给出一致的通过结果。输出仍遵守原来的自然回答或审核JSON协议。"""


def responsibility_index(documents):
    result, seen = [], set()
    heading_re = re.compile(r"(?m)^[ \t]*(#{1,6})[ \t]+([^\r\n]+)")
    for reference, doc in documents.items():
        for number, excerpt in enumerate(doc["excerpts"]):
            text = excerpt["text"]
            headings = list(heading_re.finditer(text))
            for i, heading in enumerate(headings):
                label = heading[2].strip().strip("【】 ")
                if not label.endswith("保险金") or len(label) > 60:
                    continue
                end = next(
                    (
                        h.start()
                        for h in headings[i + 1 :]
                        if len(h[1]) <= len(heading[1])
                    ),
                    len(text),
                )
                body = text[heading.end() : end]
                for clause in re.finditer(
                    r"[^。！？]*?(?:保险期间届满|延续|等待期)[^。！？]*(?:[。！？]|$)",
                    body,
                ):
                    quote = clause[0].strip()
                    key = (reference, label, quote)
                    if not quote or key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        {
                            "reference": reference,
                            "excerpt": number,
                            "responsibility_heading": heading[0].strip(),
                            "heading_span": [heading.start(), heading.end()],
                            "clause_span": [
                                heading.end() + clause.start(),
                                heading.end() + clause.end(),
                            ],
                            "verbatim_clause": quote,
                        }
                    )
    tokenizer = tiktoken.get_encoding("o200k_base")
    retained, used = [], 0
    for row in result:
        size = len(tokenizer.encode(json.dumps(row, ensure_ascii=False)))
        if len(retained) < 32 and used + size <= 2000:
            retained.append(row)
            used += size
    return {
        "entries": retained,
        "omitted": len(result) - len(retained),
        "tokens": used,
        "token_budget": 2000,
        "basis": "exclusive original Markdown heading intervals in the same-request excerpts",
    }


def boundary_conflicts(question, answer, documents):
    compact = lambda text: re.sub(r"[\s,*_`]|，", "", text)
    boundary = re.compile(
        r"(?:刚好|恰好|正好|达到)(\d+(?:\.\d+)?)(万元|元|周岁|日|年|%)"
    )
    requested = set(boundary.findall(compact(question)))
    if not requested or not re.search(
        r"是的[，,\s]*(?:必须|需要|应当)[^。\n]{0,15}(?:体检|提交|提供)", answer
    ):
        return []
    source = compact(
        "\n".join(x["text"] for d in documents.values() for x in d["excerpts"])
    )
    strict = set(re.findall(r"超过(\d+(?:\.\d+)?)(万元|元|周岁|日|年|%)", source))
    return [
        f"题目只达到{number}{unit}，同次原文金额条件为超过{number}{unit}，答案却无条件肯定必须办理；须区分等于和严格超过。"
        for number, unit in sorted(requested & strict)
    ]


def plan_conflicts(question, answer, documents, scopes):
    """Reject a positive benefit assigned to an explicit unsupported plan.

    Absence never generates a non-coverage conclusion. The check only refuses
    to transfer a benefit from another explicit plan in the same source.
    """
    compact = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    plans = set(re.findall(r"(?:保障)?计划([一二三四五六七八九十0-9]+)", question))
    attributes = [
        name for name in ["身故", "高度残疾", "住院津贴", "医疗"] if name in question
    ]
    if not plans or not attributes:
        return []
    clean_answer = re.sub(r"[*`]", "", answer)
    conflicts = set()
    for scope in scopes:
        number = re.search(r"计划([一二三四五六七八九十0-9]+)", scope["plan"])
        doc = documents[scope["reference"]]
        product = doc.get("product_name")
        if not number or number[1] not in plans or not product:
            continue
        # Inherited fragments may be incomplete; only an explicit plan block
        # can support this conservative refusal of an affirmative formula.
        if not re.search(
            r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?保障计划" + number[1], scope["text"]
        ):
            continue
        blocks = re.split(r"(?m)^(?=(?:#{1,6}[ \t]+|\d+[.、][ \t]+))", clean_answer)
        for block in blocks:
            first = block.splitlines()[0] if block.splitlines() else ""
            if "计划" + number[1] not in first or compact(product) not in compact(
                first
            ):
                continue
            if re.search(
                r"未列|未包含|未提供|仅列|只列|不能将|不能套用|不可套用|无明确", block
            ):
                continue
            if not re.search(
                r"(?:给付金额|保险金(?:的)?(?:金额|计算(?:方式|方法|公式)?|给付(?:金额|标准|方式)?)?)[：:，,\s]*(?:按|为|是|取|等于)",
                block,
            ):
                continue
            for attribute in attributes:
                siblings = [
                    other
                    for other in scopes
                    if other["reference"] == scope["reference"]
                    and other["plan"] != scope["plan"]
                ]
                if attribute not in scope["text"] and any(
                    attribute in other["text"] for other in siblings
                ):
                    conflicts.add(
                        f"{product}的{scope['plan']}原文区间未列明{attribute}给付，{attribute}文字在同来源其他计划区间；答案却在该计划下断言给付金额。不得移用其他计划公式，应说明本次计划清单与依据范围。"
                    )
    return sorted(conflicts)


def plan_heading_index(documents, scopes):
    """Expose only verbatim benefit headings inside explicit plan intervals."""
    result = []
    for scope in scopes:
        headings = []
        for line in scope["text"].splitlines():
            label = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", line).strip().strip("【】 ")
            if len(label) <= 40 and label.endswith("保险金"):
                headings.append(line)
        result.append(
            {
                "reference": scope["reference"],
                "product_name": documents[scope["reference"]].get("product_name"),
                "plan": scope["plan"],
                "verbatim_headings_in_original_interval": headings,
            }
        )
    return result


def guarded_provider(settings, provider, record):
    scopes = contract_module(settings, "source_scope").plan_scope_index

    async def complete(prompt, system_prompt=None, **kwargs):
        request = None
        try:
            value = json.loads(prompt)
            if (
                isinstance(value, dict)
                and {"question", "answer", "source_documents"} <= value.keys()
            ):
                request = value
        except (TypeError, ValueError):
            pass
        if request is not None:
            documents = request["source_documents"]
            source_scopes = scopes(documents)
            conflicts = boundary_conflicts(
                request["question"], request["answer"], documents
            ) + plan_conflicts(
                request["question"], request["answer"], documents, source_scopes
            )
            request.update(
                plan_scope_index=source_scopes,
                responsibility_scope_index=responsibility_index(documents),
                detected_shared_guard_conflicts=conflicts,
            )
            actual_prompt = json.dumps(request, ensure_ascii=False)
        else:
            match = re.search(
                r"<source_documents>\s*([\s\S]*?)\s*</source_documents>",
                system_prompt or "",
            )
            documents = json.loads(match[1]) if match else {}
            conflicts = []
            actual_prompt = prompt
        actual_system = (system_prompt or "") + "\n\n" + POLICY
        if documents:
            actual_system += (
                "\n以下仅逐字列出各计划原文区间内的保险金标题；后续免责或总体利益不补入计划清单。"
                "未列明不是无条件不保，但不得借其他计划公式回答本计划。\n<plan_responsibility_headings>\n"
                + json.dumps(
                    plan_heading_index(documents, scopes(documents)), ensure_ascii=False
                )
                + "\n</plan_responsibility_headings>"
            )
        if documents and request is None:
            actual_system += (
                "\n<responsibility_scope_index>\n"
                + json.dumps(responsibility_index(documents), ensure_ascii=False)
                + "\n</responsibility_scope_index>"
            )
        record(
            {
                "stage": "shared_answer_guard",
                "state": "provider_started",
                "contract": VERSION,
                "audit": request is not None,
                "prompt": actual_prompt,
                "system_prompt": actual_system,
                "input_sha256": digest(actual_system + "\n" + actual_prompt),
            }
        )
        raw = await provider(actual_prompt, system_prompt=actual_system, **kwargs)
        if request is not None and conflicts and isinstance(raw, str):
            try:
                verdict = json.loads(raw)
            except ValueError:
                return raw
            if all(
                verdict.get(k) is True
                for k in ["supported", "scope_consistent", "formula_consistent"]
            ):
                rejected = {
                    "supported": False,
                    "scope_consistent": False,
                    "formula_consistent": False,
                    "issues": conflicts,
                    "corrected_answer": None,
                }
                record(
                    {
                        "stage": "shared_answer_guard",
                        "state": "verdict_rejected",
                        "provider_raw_response": raw,
                        "returned_verdict": rejected,
                    }
                )
                return json.dumps(rejected, ensure_ascii=False)
        if request is not None and isinstance(raw, str):
            try:
                verdict = json.loads(raw)
            except ValueError:
                return raw
            issues = verdict.get("issues", [])
            debate = r"让我|等等|难道|那为什么|为什么我要|重新审视|可能是测试|看起来答案|错误在哪里|那错误在哪|似乎应该|我是否|我需要"
            if isinstance(issues, list) and (
                len(issues) > 8
                or any(
                    isinstance(x, str) and (len(x) > 600 or re.search(debate, x))
                    for x in issues
                )
            ):
                error = {
                    "protocol_error": "审核输出包含自我辩论或超长非事实说明；只列具体原文反例，没有具体错误时返回一致通过，不能用常识推定未列责任。"
                }
                record(
                    {
                        "stage": "shared_answer_guard",
                        "state": "audit_protocol_rejected",
                        "provider_raw_response": raw,
                        "returned_error": error,
                    }
                )
                # The existing strict schema invokes its bounded protocol
                # recovery. Never reinterpret a negative as a positive.
                return json.dumps(error, ensure_ascii=False)
        return raw

    return complete

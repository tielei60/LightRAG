"""Source-derived answer obligations and deliberately narrow delivery checks."""

from __future__ import annotations

import re
import unicodedata

_GENERIC_NAME_PARTS = (
    "保险",
    "寿险",
    "人寿",
    "保单",
    "合同",
    "产品",
    "尊享版",
    "终身",
    "年金",
    "医疗",
    "意外",
    "伤害",
    "综合",
    "重疾",
    "养老",
    "定期",
    "两全",
    "分红型",
    "万能型",
)
_AGE_POINT = r"\d+周岁(?:的)?保单周年日(?:[（(]见释义[）)])?零时(?:之)?[前后]"
_PAYMENT_POINT = (
    r"最后(?:一个|一)?保险费约定支付日后(?:的)?首个保单周年日零时(?:之)?[前后]"
)
_GROWTH_FORMULA = (
    r"基本保险金额[×xX*]\d+(?:\.\d+)?(?:\^?[（(]?)N[-−]1[）)]?[^。；]{0,70}"
)
_FREE_LOOK = r"自(?:您)?签收[^。；]{0,25}之日起[^。；]{0,15}?(\d+)日的?犹豫期"
_BUSINESS_ASSERTION = re.compile(
    r"(?:需要|体检|给付|赔付|保证|领取|退还|承担|豁免|门槛|费率|基本保额|计算基数)"
)
_NON_ASSERTION = re.compile(
    r"(?:不讨论|未询问|未要求|不涉及|无法确认|不能据此|不能推断)"
)
_LOSS_WARNING = r"(?:犹豫期后解除(?:本)?合同|退保)[^。；]{0,25}可能[^。；]{0,20}损失"
_PAID_PREMIUM = r"(?:累计)?(?:已[交缴付]|已支付|所[交缴付])(?:保险费|保费)"
_VALUE_OPERATORS = {
    "不低于": "ge",
    "不少于": "ge",
    "不高于": "le",
    "不多于": "le",
    "低于": "lt",
    "小于": "lt",
    "少于": "lt",
    "不足": "lt",
    "不及": "lt",
    "高于": "gt",
    "大于": "gt",
    "多于": "gt",
    "超过": "gt",
    "等于": "eq",
}
_VALUE_MODAL = re.compile(r"通常|一般|往往|必然|一定|必定|总是|肯定")
_VALUE_COMPARISON_NEGATED = re.compile(
    r"不(?:能|可|应)(?:据此)?(?:断言|认定|推断|说|保证)|无法(?:断言|确认|认定|判断)"
    r"|不一定|未必|并非|并不是|不是说|不能确定|没有依据|不代表|不意味着"
    r"|(?:没有|未|不)(?:据此)?(?:说明|声称|证明|支持)"
)


def _qualified_value_comparisons(text):
    """Recognize affirmative frequency/necessity comparisons, not probability."""
    operators = "|".join(_VALUE_OPERATORS)
    for sentence in re.split(r"[。！？；;，,:：]|但是|不过|然而|但", text):
        compact = _compact(re.sub(r"[*_`]", "", sentence))
        for reverse in (False, True):
            left, right = (
                (_PAID_PREMIUM, "现金价值") if reverse else ("现金价值", _PAID_PREMIUM)
            )
            pattern = (
                rf"{left}[^。；]{{0,20}}?(?P<op>{operators})[^。；]{{0,15}}?{right}"
            )
            for match in re.finditer(pattern, compact):
                before_operator = compact[
                    max(0, match.start() - 12) : match.start("op")
                ]
                modal = _VALUE_MODAL.search(before_operator)
                context = compact[max(0, match.start() - 18) : match.end()]
                if not modal or _VALUE_COMPARISON_NEGATED.search(context):
                    continue
                relation = _VALUE_OPERATORS[match["op"]]
                if reverse:
                    relation = {
                        "lt": "gt",
                        "gt": "lt",
                        "le": "ge",
                        "ge": "le",
                        "eq": "eq",
                    }[relation]
                yield {
                    "text": sentence,
                    "relation": relation,
                    "modality": "certain"
                    if modal[0] in {"必然", "一定", "必定", "总是", "肯定"}
                    else "typical",
                }


def _compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace("−", "-")


def _product_mentions(text, products, *, minimum=4):
    """Match literal full names or unique, non-generic source-name spans.

    Four characters establish a conservative match; shorter spans only identify
    uncertainty and never authorize a hard scope rejection. Pronouns stay with
    the model. Generic insurer/type/version words never establish a short alias.
    """
    normalized = _compact(text)
    names = {name: _compact(name) for name in products}
    owners = {}
    for name, value in names.items():
        generic_offsets = {
            offset
            for part in _GENERIC_NAME_PARTS
            for match in re.finditer(re.escape(part), value)
            for offset in range(match.start(), match.end())
        }
        for size in range(minimum, 5):
            for offset in range(len(value) - size + 1):
                token = value[offset : offset + size]
                if generic_offsets.intersection(range(offset, offset + size)):
                    continue
                owners.setdefault(token, set()).add(name)
    found = {
        name
        for name, value in names.items()
        if value in normalized
        or any(
            token in normalized and owned == {name} for token, owned in owners.items()
        )
    }
    return sorted(
        name
        for name in found
        if not any(names[name] in names[other] for other in found if other != name)
    )


def _source_matches(text, pattern):
    # Keep source spelling and offsets, including PDF line breaks.
    offsets = [index for index, character in enumerate(text) if not character.isspace()]
    compact = "".join(text[index] for index in offsets)
    for match in re.finditer(pattern, compact):
        start, end = offsets[match.start()], offsets[match.end() - 1] + 1
        yield text[start:end], match


def source_answer_requirements(question, documents, history_messages=None):
    """Expose exact relevant premises without removing or rewriting any source."""
    products = sorted(
        {
            document["product_name"]
            for document in documents.values()
            if document["product_name"]
        }
    )
    requested = _product_mentions(question, products)
    uncertain = sorted(
        set(_product_mentions(question, products, minimum=2)) - set(requested)
    )
    if re.search(r"它|这个|那个|前面|上面|上述|之前|前述|两款|两种|两者", question):
        history = "\n".join(
            message["content"]
            for message in history_messages or []
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
        uncertain = sorted(
            set(uncertain)
            | (set(_product_mentions(history, products)) - set(requested))
        )
    facts = []
    seen = set()
    formulas_relevant = bool(
        re.search(r"公式|增长|因子|几项|两项|三项|阶段|什么时候|何时", question)
    )
    free_look_relevant = "犹豫期" in question and bool(
        re.search(r"退保|解除|退还|退款|退回|返还", question)
    )
    basis_relevant = bool(re.search(r"累计|体检|财务|限额|门槛", question))
    surrender_relevant = bool(re.search(r"退保|解除(?:本)?合同", question))
    survival_relevant = "生存保险金" in question and bool(
        re.search(r"开始领|领到|领取时间|什么时候|何时|给付比例", question)
    )
    care_relevant = "护理状态" in question and bool(
        re.search(r"相同|一样|不同|区别|差异|比较|具体要求", question)
    )
    for reference, document in documents.items():
        product = document["product_name"]
        for index, excerpt in enumerate(document["excerpts"]):
            text = excerpt["text"]
            patterns = []
            if formulas_relevant and product in requested:
                patterns += [
                    ("age_anniversary", _AGE_POINT),
                    ("payment_anniversary", _PAYMENT_POINT),
                    ("growth_formula", _GROWTH_FORMULA),
                ]
            if free_look_relevant and product in requested:
                patterns.append(("free_look_definition", _FREE_LOOK))
            if surrender_relevant and product in requested:
                patterns.append(("surrender_loss_warning", _LOSS_WARNING))
                if re.search(
                    r"手续|资料|材料|终止|时点|何时解除|什么时候解除", question
                ):
                    patterns.append(
                        (
                            "termination_event",
                            r"自我们收到解除合同通知书时起[，,]本合同终止",
                        )
                    )
            if survival_relevant and product in requested:
                patterns.append(
                    ("survival_condition", r"被保险人在每个保单周年日零时生存")
                )
            if care_relevant and product in requested:
                patterns += [
                    ("care_dependence", r"日常生活持续依赖他人监护和照顾"),
                    (
                        "irreversibility_definition",
                        r"永久不可逆[：:]指自疾病确诊或意外伤害发生之日起[，,]经过积极治疗\d+天后[，,]仍无法通过现有医疗手段恢复",
                    ),
                ]
            if (
                product in requested
                and "身故" in question
                and re.search(r"计算|怎么算|公式|金额|给付|赔付", question)
                and not re.search(r"增长因子|增长系数", question)
            ):
                patterns.append(("death_contract_termination", r"身故[，,]本合同终止"))
            if basis_relevant:
                patterns.append(
                    ("cumulative_basis", r"累计[^。；\n]{0,70}[＝=][^。；\n]{0,140}")
                )
            for kind, pattern in patterns:
                for quote, match in _source_matches(text, pattern):
                    identity = (kind, reference, product, _compact(quote))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    fact = {
                        "kind": kind,
                        "reference": reference,
                        "product_name": product,
                        "excerpt": index,
                        "text": quote,
                    }
                    if kind == "free_look_definition":
                        fact["days"] = int(match[1])
                    facts.append(fact)
            if surrender_relevant and product in requested:
                for comparison in _qualified_value_comparisons(text):
                    identity = (
                        "surrender_value_comparison",
                        reference,
                        product,
                        _compact(comparison["text"]),
                    )
                    if identity not in seen:
                        seen.add(identity)
                        facts.append(
                            {
                                "kind": "surrender_value_comparison",
                                "reference": reference,
                                "product_name": product,
                                "excerpt": index,
                                **comparison,
                            }
                        )
    return {
        "requested_products": requested,
        "possible_short_name_products": uncertain,
        "product_scope_resolved": bool(requested) and not uncertain,
        "available_products": products,
        "source_facts": facts,
    }


def _chinese_integer(number):
    digits = "零一二三四五六七八九"
    if 0 < number < 10:
        return digits[number]
    if 10 <= number < 100:
        tens, units = divmod(number, 10)
        return (
            (digits[tens] if tens > 1 else "") + "十" + (digits[units] if units else "")
        )
    return str(number)


def answer_requirement_conflicts(answer, requirements):
    """Catch only literal scope/precision defects supported by current sources.

    These guards do not prove semantic correctness. All repairs remain authored
    by the configured model and are independently reviewed before delivery.
    """
    conflicts = []
    requested = set(requirements["requested_products"])
    if not requested:
        return conflicts
    products = requirements["available_products"]
    for sentence in re.split(r"[。！？\n]", answer):
        if not _BUSINESS_ASSERTION.search(sentence) or _NON_ASSERTION.search(sentence):
            continue
        for product in (
            set(_product_mentions(sentence, products))
            - requested
            - set(requirements.get("possible_short_name_products", []))
        ):
            message = f"答案对未询问产品{product}作出额外业务断言；检索到该产品不等于问题参数适用于它。"
            if message not in conflicts:
                conflicts.append(message)
    facts = requirements["source_facts"]
    compact = _compact(re.sub(r"[*_`]", "", answer))
    describes_phase = re.search(
        r"\d+周岁[^。；\n]{0,28}(?:身故|高度残疾|之前|之后|前|后|阶段)"
        r"|[缴交]费期满(?:前|后)"
        r"|(?:两项|三项|两者|三者|(?:两|三|二|2|3)个比较项)[^。；\n]{0,20}(?:取大|较大|最大)",
        compact,
    )
    if describes_phase:
        for kind, label, pattern in (
            (
                "age_anniversary",
                "年龄对应的保单周年日零时界点",
                r"\d+周岁(?:的|对应的|所对应的)?保单周年日(?:\([^)]*\))?(?:的)?(?:零时|零点|0:00|00:00)",
            ),
            (
                "payment_anniversary",
                "最后约定支付日后的首个保单周年日零时界点",
                r"最后(?:一个|一期|一)?(?:保险费|保费)约定支付日(?:之)?后(?:的)?(?:首个|第一个)保单周年日(?:的)?(?:零时|零点|0:00|00:00)",
            ),
        ):
            if any(row["kind"] == kind for row in facts) and not re.search(
                pattern, compact
            ):
                conflicts.append(
                    f"答案说明给付阶段时遗漏原文明示的{label}，不能以生日或交完费当天替代。"
                )
    definitions = [row for row in facts if row["kind"] == "free_look_definition"]
    days = {row["days"] for row in definitions}
    if len(days) == 1:
        number = next(iter(days))
        duration = rf"(?:{number}|{_chinese_integer(number)})(?:日|天)"
        if "签收" not in compact or not re.search(duration, compact):
            conflicts.append(
                "答案说明犹豫期内外退保时遗漏同次原文明示的签收起算点或犹豫期时长。"
            )
    if any(row["kind"] == "growth_formula" for row in facts):
        short_formula = re.search(
            r"基本保(?:险金|险|金)?额[×xX*]增长(?:因子|系数)([^。；\n]{0,35})", compact
        )
        factor_is_power = re.search(
            r"增长(?:因子|系数)(?:为|是|=|:)[^。；\n]{0,30}N-1", compact
        )
        if short_formula and "N-1" not in short_formula[1] and not factor_is_power:
            conflicts.append(
                "答案重述基本保额增长公式时省略原文指数，不能将复利幂项缩成基本保额乘增长因子。"
            )
        if re.search(r"N[-−]1", compact) and not re.search(
            r"N(?:(?:的)?(?:含义|定义))?(?:为|是|代表|指|表示|=|:)[^。；]{0,40}保单年度"
            r"|N[（(][^）)]{0,35}保单年度",
            compact,
        ):
            conflicts.append("答案写出了含N的给付公式，却未交代原文明示的N含义。")
    kinds = {row["kind"] for row in facts}
    if "death_contract_termination" in kinds and not re.search(
        r"合同(?:效力)?(?:即|随之|同时|相应|也|将|会|自动)?终止|终止(?:本)?合同",
        compact,
    ):
        conflicts.append(
            "身故保险金给付说明遗漏同次原文明示的本合同终止结果，须随该项给付说明一起交代。"
        )
    if "survival_condition" in kinds and not re.search(
        r"(?:每个|各个|每一|每年)[^。；]{0,12}保单周年日(?:的)?(?:零时|零点|0:00|00:00)[^。；]{0,12}生存",
        compact,
    ):
        conflicts.append(
            "生存保险金给付须说明原文明示的被保险人在每个保单周年日零时生存条件。"
        )
    if "termination_event" in kinds and not re.search(
        r"收到(?:您(?:的)?)?解除(?:本)?合同(?:的)?通知书[^。；]{0,30}(?:合同[^。；]{0,10}(?:终止|解除)|(?:终止|解除)[^。；]{0,10}合同)"
        r"|合同[^。；]{0,10}(?:于|自|在)[^。；]{0,15}收到(?:您(?:的)?)?解除(?:本)?合同(?:的)?通知书[^。；]{0,10}(?:终止|解除)",
        compact,
    ):
        conflicts.append(
            "解除手续须说明保险公司收到解除合同通知书时合同终止，不能用退款时限替代终止时点。"
        )
    if "care_dependence" in kinds and not re.search(
        r"(?:日常生活)?[^。；]{0,10}持续[^。；]{0,15}他人[^。；]{0,10}监护[^。；]{0,10}照顾",
        compact,
    ):
        conflicts.append(
            "护理状态比较遗漏原文明示的日常生活持续依赖他人监护和照顾条件。"
        )
    for fact in facts:
        if fact["kind"] == "irreversibility_definition" and "永久不可逆" in compact:
            days_match = re.search(r"(\d+)天", _compact(fact["text"]))
            if days_match and not re.search(
                rf"(?:治疗)[^。；]{{0,8}}{days_match[1]}(?:天|日)[^。；]{{0,35}}(?:恢复|不可逆)",
                compact,
            ):
                conflicts.append(
                    "护理状态中的永久不可逆须解释同次原文明示的起算事件、积极治疗期限及仍无法恢复的含义。"
                )
                break
    warned_products = {
        row["product_name"] for row in facts if row["kind"] == "surrender_loss_warning"
    }
    for comparison in _qualified_value_comparisons(answer):
        subjects = (
            requested
            if len(requested) == 1
            else set(_product_mentions(comparison["text"], products)) & requested
        )
        for product in subjects & warned_products:
            supported = any(
                row["kind"] == "surrender_value_comparison"
                and row["product_name"] == product
                and row["relation"] == comparison["relation"]
                and (
                    row["modality"] == comparison["modality"]
                    or row["modality"] == "certain"
                )
                for row in facts
            )
            if not supported:
                message = f"{product}的原文提示退保可能损失，未支持将现金价值与已交保费的比较强化为通常或必然；应保留原文模态。"
                if message not in conflicts:
                    conflicts.append(message)
    return conflicts

"""Conservative guard for explicit fee-base equivalence claims.

This is not a general mathematical/semantic verifier. It flags an affirmative
same-base claim when the same source defines two named fees with different
literal bases. Negations, questions and rate-only comparisons are not claims.
"""

import re
from itertools import combinations

_FORMULA = re.compile(
    r"([\u4e00-\u9fff]{2,20}费用)[=＝]([^=＝×*。；;]{1,100})[×*]([\u4e00-\u9fff]{2,24}(?:费用率|费率))"
)
_EQUIVALENCE = re.compile(r"基数[^。；;？！?\n]{0,20}?(?:相同|一致|一样|等同)")
_NEGATION = re.compile(r"不|并非|不能|未必|无依据|错误|误称")


def fee_formula_index(documents: dict) -> list[dict]:
    rows = []
    seen = set()
    for reference, document in documents.items():
        for excerpt in document["excerpts"]:
            for line in excerpt["text"].splitlines():
                normalized = re.sub(r"\s+", "", line)
                for match in _FORMULA.finditer(normalized):
                    label, base, rate = match.groups()
                    key = (reference, label, base, rate)
                    if key not in seen:
                        seen.add(key)
                        rows.append(
                            {
                                "reference": reference,
                                "fee": label,
                                "base": base,
                                "rate": rate,
                            }
                        )
    return rows


def formula_conflicts(answer: str, formulas: list[dict]) -> list[str]:
    claims = []
    for sentence in re.split(r"[。；;\n]", answer):
        if "?" in sentence or "？" in sentence:
            continue
        plain = re.sub(r"[*_`\s]", "", sentence)
        for match in _EQUIVALENCE.finditer(plain):
            # Keep the check local: a later contradictory explanation does not
            # cancel an earlier affirmative claim in another sentence.
            window = plain[max(0, match.start() - 12) : match.end()]
            if not _NEGATION.search(window):
                claims.append(match.group())
    if not claims:
        return []
    errors = []
    for left, right in combinations(formulas, 2):
        if (
            left["reference"] == right["reference"]
            and left["fee"] != right["fee"]
            and left["base"] != right["base"]
            and left["fee"] in answer
            and right["fee"] in answer
        ):
            errors.append(
                f"原文{left['reference']}的{left['fee']}基数为「{left['base']}」，"
                f"{right['fee']}基数为「{right['base']}」。答案却声称「{claims[0]}」。"
                "不同原文定义不能未经证据直接等同；请保留两个基数及引用并修正比较结论。"
            )
    return list(dict.fromkeys(errors))

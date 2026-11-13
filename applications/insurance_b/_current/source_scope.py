"""Locate explicit plan headings without inferring contractual responsibilities."""

import re

_PLAN = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(保障计划[一二三四五六七八九十0-9]+)[：:]?[ \t]*$"
)
_SECTION = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\d+\.\d+[ \t]+|[（(][一二三四五六七八九十]+[）)][ \t]*(?=责任免除|投保范围|保险期间|交费方式|保单利益|等待期|利益演示|犹豫期|退保))")


def plan_scope_index(documents):
    """Verbatim spans; a missing mention is not a proof of non-coverage.

    Multi-plan bodies are split only at explicit heading boundaries. Standalone
    excerpts inherit a plan only from an unambiguous governing heading. Never
    assign all children to the last plan in a multi-plan parent.
    """
    scopes = []
    for reference, document in documents.items():
        for index, excerpt in enumerate(document["excerpts"]):
            text = excerpt["text"]
            headings = list(_PLAN.finditer(text))
            if headings:
                for offset, heading in enumerate(headings):
                    end = (
                        headings[offset + 1].start()
                        if offset + 1 < len(headings)
                        else len(text)
                    )
                    section = _SECTION.search(text, heading.end(), end)
                    if section:
                        end = section.start()
                    scopes.append(
                        {
                            "reference": reference,
                            "excerpt": index,
                            "plan": heading[1],
                            "start": heading.start(),
                            "end": end,
                            "text": text[heading.start() : end],
                        }
                    )
            else:
                parents = [
                    match[1]
                    for parent in excerpt["governing_context"]
                    for match in _PLAN.finditer(parent)
                ]
                if len(set(parents)) == 1:
                    scopes.append(
                        {
                            "reference": reference,
                            "excerpt": index,
                            "plan": parents[0],
                            "start": 0,
                            "end": len(text),
                            "text": text,
                        }
                    )
    return scopes

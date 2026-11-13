"""Publish descriptive paired results without turning model review into acceptance."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ..common import digest, read_json, save_json
from ..config import PROJECT_ROOT, ROOT


def provider_usage(run_dir):
    frozen = run_dir / "experiment-provider-events.jsonl"
    if not frozen.exists():
        with frozen.open("xb") as stream:
            stream.write((run_dir / "provider-events.jsonl").read_bytes())
    events = [json.loads(line) for line in frozen.read_text().splitlines()]
    roles = defaultdict(
        lambda: {
            "wrapper_calls": 0,
            "terminal_observations": 0,
            "usage_records": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    for event in events:
        row = roles[event["role"]]
        if event["state"] == "started":
            row["wrapper_calls"] += 1
        if (
            event["state"] == "stream_consumed"
            or event["state"] == "response_received"
            and not event.get("stream")
        ):
            row["terminal_observations"] += 1
            for usage in event.get("usage", []):
                row["usage_records"] += 1
                for name in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    row[name] += usage.get(name, 0)
    return dict(roles)


def latency(rows):
    seconds = sorted(r["total_seconds"] for r in rows)
    return {
        "requests_including_failures": len(seconds),
        "total_seconds": sum(seconds),
        "p50_seconds": statistics.median(seconds),
        "p95_seconds": seconds[math.ceil(len(seconds) * 0.95) - 1],
        "max_seconds": max(seconds),
    }


def actual_prompt_sizes(rows):
    import tiktoken

    tokenizer = tiktoken.get_encoding("o200k_base")
    stages = defaultdict(list)
    for row in rows:
        for event in row.get("same_request_events", []):
            if (
                event.get("stage") == "shared_answer_guard"
                and event.get("state") == "provider_started"
            ):
                text = event["system_prompt"] + "\n" + event["prompt"]
                stages["audit" if event["audit"] else "draft"].append(
                    len(tokenizer.encode(text))
                )
    return {
        "kind": "o200k_base tokenizer estimate of actual prompt/system text; excludes chat framing",
        "stages": {
            name: {
                "calls": len(values),
                "p50_tokens": statistics.median(values),
                "max_tokens": max(values),
                "total_tokens": sum(values),
            }
            for name, values in stages.items()
        },
    }


def publish(a_path, b_path, review_dir, destination):
    a, b = read_json(a_path), read_json(b_path)
    fields = [
        "question_set_sha256",
        "source_set_sha256",
        "query_contract_sha256",
        "extraction_contract_sha256",
    ]
    if any(not a.get(k) or a[k] != b.get(k) for k in fields):
        raise ValueError("Main experiment controls differ")
    key = lambda r: (r["case_id"], r["repeat"])
    runs = {"A": a, "B": b}
    results = {arm: {key(r): r for r in run["results"]} for arm, run in runs.items()}
    if len(results["A"]) != 87 or results["A"].keys() != results["B"].keys():
        raise ValueError("Expected 29 matched questions in three rounds")
    auxiliary = read_json(review_dir / "auxiliary-review.json")
    reviews = {(r["arm"], *key(r)): r for r in auxiliary["rows"]}
    if len(reviews) != 174:
        raise ValueError("Blind review is incomplete")
    pairs, bins = [], Counter()
    by_question = defaultdict(list)
    for identifier in sorted(results["A"]):
        values = {}
        for arm in ["A", "B"]:
            raw, review = results[arm][identifier], reviews[(arm, *identifier)]
            values[arm] = bool(raw["delivered"] and review["full_answer_passed"])
        cell = (
            "双方通过"
            if all(values.values())
            else "A未通过/B通过"
            if values["B"]
            else "A通过/B未通过"
            if values["A"]
            else "双方未通过"
        )
        bins[cell] += 1
        row = {
            "case_id": identifier[0],
            "repeat": identifier[1],
            "cell": cell,
            "auxiliary_pass": values,
        }
        pairs.append(row)
        by_question[identifier[0]].append(row)
    stats = {}
    for arm, run in runs.items():
        raw = run["results"]
        judgments = [reviews[(arm, *key(row))] for row in raw]
        stats[arm] = {
            "variant": run["variant"],
            "delivered": sum(r["delivered"] for r in raw),
            "auxiliary_passed": sum(
                bool(r["delivered"] and j["full_answer_passed"])
                for r, j in zip(raw, judgments, strict=True)
            ),
            "stable_questions_3_of_3": sum(
                all(p["auxiliary_pass"][arm] for p in rows)
                for rows in by_question.values()
            ),
            "high_risk_error_answers": sum(j["high_risk_error"] for j in judgments),
            "latency": latency(raw),
            "actual_prompt_sizes": actual_prompt_sizes(raw),
            "provider_usage": provider_usage(
                a_path.parents[2] if arm == "A" else b_path.parents[2]
            ),
        }
    baseline = read_json(ROOT / "snapshots/a-before-b1.json")
    a_changes = [
        name
        for name, sha in baseline["a_code_hashes"].items()
        if digest((PROJECT_ROOT / name).read_bytes()) != sha
    ]
    for name, sha in baseline["profiles"].items():
        if (
            digest(
                (
                    PROJECT_ROOT / "applications/insurance/prompt_profiles" / name
                ).read_bytes()
            )
            != sha
        ):
            a_changes.append("profile:" + name)
    registry_unchanged = (
        digest((PROJECT_ROOT / "output/workspaces/master/registry.json").read_bytes())
        == baseline["a_registry_sha256"]
    )
    copied_config = read_json(ROOT / ".runtime/model-config-import.json")
    project_env_unchanged = (
        digest((PROJECT_ROOT / ".env").read_bytes()) == copied_config["source_sha256"]
    )
    summary = {
        "formal_business_acceptance": False,
        "review_method": "blind model-assisted original-source review; human review pending",
        "unique_questions": 29,
        "statistical_independence": "not assumed: questions share products, sources and some regression facts",
        "paired_repeated_requests": 87,
        "four_cells": dict(bins),
        "statistics": stats,
        "pairs": pairs,
        "a_live_source_changes": a_changes,
        "a_registry_unchanged": registry_unchanged,
        "a_project_env_unchanged": project_env_unchanged,
        "controls": {k: a[k] for k in fields},
        "report_code_sha256": digest(Path(__file__).read_bytes()),
        "comparison_kind": "development whole-chain comparison; not a strict isolated preparation/P experiment",
        "retry_policy_difference": "A preflight max3 selections per input; one B input required3 native processing invocations and9 selections. All attempts retained in provider usage.",
    }
    adjudication_path = destination / "source-adjudication.json"
    adjudication = read_json(adjudication_path) if adjudication_path.exists() else None
    summary["source_adjudication"] = adjudication
    integrity_path = destination / "integrity-report.json"
    service_path = destination / "service-status.json"
    summary["engineering"] = (
        read_json(integrity_path) if integrity_path.exists() else None
    )
    summary["local_B_service"] = (
        read_json(service_path) if service_path.exists() else None
    )
    save_json(destination / "paired-summary.json", summary)
    documents = {}
    for arm, path in [("A", a_path), ("B", b_path)]:
        documents[arm] = read_json(path.parents[2] / "readback-report.json")
    table = [
        "| 指标 | A* 现有业务预分块 | B1 结构解析 + 原生 P |",
        "| --- | ---: | ---: |",
    ]
    for label, left, right in [
        ("资料/产品", "21 / 7", "21 / 7"),
        ("原生读回", documents["A"]["passed"], documents["B"]["passed"]),
        (
            "实际索引片段（规模诊断）",
            documents["A"]["counts"]["chunks"],
            documents["B"]["counts"]["chunks"],
        ),
        (
            "原生实体 / 关系（规模诊断）",
            f"{documents['A']['counts']['entities']} / {documents['A']['counts']['relationships']}",
            f"{documents['B']['counts']['entities']} / {documents['B']['counts']['relationships']}",
        ),
        (
            "交付成功（含失败分母87）",
            f"{stats['A']['delivered']}/87",
            f"{stats['B']['delivered']}/87",
        ),
        (
            "辅助核验整题通过",
            f"{stats['A']['auxiliary_passed']}/87",
            f"{stats['B']['auxiliary_passed']}/87",
        ),
        (
            "三轮均通过的不同题",
            f"{stats['A']['stable_questions_3_of_3']}/29",
            f"{stats['B']['stable_questions_3_of_3']}/29",
        ),
        (
            "辅助核验高风险错误答案",
            stats["A"]["high_risk_error_answers"],
            stats["B"]["high_risk_error_answers"],
        ),
        (
            "请求 P50 / P95（秒，含失败）",
            f"{stats['A']['latency']['p50_seconds']:.2f} / {stats['A']['latency']['p95_seconds']:.2f}",
            f"{stats['B']['latency']['p50_seconds']:.2f} / {stats['B']['latency']['p95_seconds']:.2f}",
        ),
    ]:
        table.append(f"| {label} | {left} | {right} |")
    lines = [
        "# 保险 B 修复与 A/B 开发集对照",
        "",
        "本报告比较同一批7产品、21份原件和29道原题，每题独立回答3次。87次重复不是87个独立样本；29道不同问题也有同源、同事实回归题，不能假定完全统计独立。全部问题已用于开发，不是未见验收集。模型辅助核验不替代业务人员验收。",
        "",
        "A*复用当前A的563个业务片段，在B专用数据库中新建索引。A*与B1共用冻结的当前抽取/回答合同及已登记修正；当前A的代码、数据和9623服务保持原样。这里的A*不是未修正线上A的成绩，且共用修正收益不能归因于P。",
        "本轮还有执行差异：A预审最多3次选择调用；B有1份输入经3次原生处理、实际9次选择调用才通过。因此本轮是开发整链路对比，不能称为严格只比较解析/P的受控实验。全部额外调用计入用量，合同哈希一致仅证明其中已记录的基础模型、提示、校验及查询设置一致。",
        "",
        *table,
        "",
        "表内通过和高风险计数均来自模型辅助核验，已有原件复核争议；尤其高风险计数为0不能证明业务高风险错误为0。正式人工业务验收尚未完成。",
        "",
        "## 配对结果",
        "",
        "| 同题同轮结果 | 次数 |",
        "| --- | ---: |",
        *[
            f"| {name} | {bins[name]} |"
            for name in ["双方通过", "A未通过/B通过", "A通过/B未通过", "双方未通过"]
        ],
        "",
        "## 原问题的修复及边界",
        "",
        "B1保留原表头及0/空白/破折号；表格与紧邻标题、注释使用连续原文证据；用真实PDF文字位置恢复上标指数；最终P片段在哈希/向量化前绑定已核实产品身份。共用修正包括计划区间边界、材料用途校验、选择提示，以及同次来源的回答约束。原始P输出、最终索引片段、实际抽取输入分别留档。",
        "",
        "这些修改均由已保存的开发失败触发。文字PDF关闭OCR和公式增强，真实上标位置恢复没有OCR/模型调用。不得把拦截错误答案算成回答正确；审核失败仍计入分母。",
        "",
        "## 四层验收状态",
        "",
        "| 层次 | 本轮能证明的内容 | 尚不能宣称的内容 |",
        "| --- | --- | --- |",
        "| 解析/切片 | 21份文档完成原生写入和新实例读回；B逐文件保留真实P与最终片段证明 | 任意未来文件都能正确解析 |",
        "| 抽取/合并 | 片段有核准证据台账；原生图、来源和三类向量读回一致 | 独立人工关键事实召回≥95%、全部合并语义无损 |",
        "| 检索证据 | 每次请求保存实际生成来源、上下文和审核轨迹 | 必需证据组完整命中≥95%（未完成逐组人工标注） |",
        "| 回答 | 上表及逐题文件报告盲评辅助结果、失败和重复稳定性 | 正式业务验收≥95%或未来问题泛化 |",
        "",
        "## 耗时、用量与隔离",
        "",
        "| 角色 | A* 实际包装器调用 / 返回total token | B1 实际包装器调用 / 返回total token |",
        "| --- | ---: | ---: |",
        *[
            f"| {label} | {stats['A']['provider_usage'][role]['wrapper_calls']} / {stats['A']['provider_usage'][role]['total_tokens']:,} | {stats['B']['provider_usage'][role]['wrapper_calls']} / {stats['B']['provider_usage'][role]['total_tokens']:,} |"
            for role, label in [
                ("extract", "抽取及合并"),
                ("keyword", "关键词"),
                ("query", "回答及审核"),
            ]
        ],
        "",
        f"B的抽取/合并total token比A*少约{(1 - stats['B']['provider_usage']['extract']['total_tokens'] / stats['A']['provider_usage']['extract']['total_tokens']) * 100:.1f}%，回答/审核则多约{(stats['B']['provider_usage']['query']['total_tokens'] / stats['A']['provider_usage']['query']['total_tokens'] - 1) * 100:.1f}%。这些是本次工作量的实际用量差异，包含B额外恢复重试，不代表费用下降同样比例，也不归因于P单一算法。",
        "",
        f"最终生成输入的本地o200k_base估计：草稿P50为A* {stats['A']['actual_prompt_sizes']['stages']['draft']['p50_tokens']:,}、B1 {stats['B']['actual_prompt_sizes']['stages']['draft']['p50_tokens']:,} token，最大分别{stats['A']['actual_prompt_sizes']['stages']['draft']['max_tokens']:,}、{stats['B']['actual_prompt_sizes']['stages']['draft']['max_tokens']:,}。18,000是原生检索预算，不是经过回答治理后最终提供商请求的硬上限；估计不含聊天封装，与提供商tokenizer口径分开。完整估计及实际回执见paired-summary.json。",
        "",
        "双方按相同并发串行执行，关闭应用答案缓存。三轮均实际调用模型。提供商包装器调用数和返回token见paired-summary.json；这不是底层HTTP重试数。Embedding/rerank未提供可靠token账单，价格、提供商缓存命中和金额不估算。",
        "",
        "B复用14份原始PDF解析缓存，A复用已有业务准备产物；本轮不是冷解析成本实验。新索引的抽取/合并重新调用模型，未回放开发期选择。独立冷抽取重建子集三次尚未测，与同索引三轮回答分开。",
        "",
        f"A源码/提示变化：{a_changes or '无'}；A注册清单哈希保持一致：{registry_unchanged}。",
        "",
        "## 采用决定",
        "",
        "本轮保留A，不自动替换。存在未通过题、高风险错误、尚未完成的事实/证据人工核验或工程缺口时，不能仅凭平均分较高认定B可替换A。B本地入口状态及管理记录见service-status.json。",
        "",
        "逐题完整答案与独立核验说明见full-qa.md；原始结果、盲评匿名映射、原件及提供商轨迹均保留，不覆盖历史失败。",
    ]
    destination.mkdir(parents=True, exist_ok=True)
    if adjudication:
        lines.extend(
            [
                "",
                "## 原件复核与评分争议",
                "",
                "以下补充核对独立于原始模型盲评，未自动覆盖其分数。辅助通过率不能作为真实业务通过率或直接选型依据。",
                "",
            ]
        )
        for item in adjudication["items"]:
            lines.extend(
                [f"**{item['case_id']}**：{item['finding']}", "", item["decision"], ""]
            )
    if summary["engineering"]:
        check = summary["engineering"]
        lines.extend(
            [
                "",
                "## 最终工程复查",
                "",
                f"B应用单元测试{check['unit_tests_passed']}项通过；合成集成检查{check['synthetic_checks_passed']}项通过，外部模型调用为0。两组测试后新进程读回通过；B重复导入复用21份原件，新增模型调用{check['B_duplicate_model_calls']}，原生计数保持一致。",
                "",
                "当前A源码、提示、注册清单及项目.env哈希均保持不变，Git变更仅位于applications/insurance_b/。",
            ]
        )
    if summary["local_B_service"]:
        service = summary["local_B_service"]
        lines.extend(
            [
                "",
                f"B本地验证入口为http://127.0.0.1:9625，workspace={service['workspace']}。健康检查、docs及OpenAPI通过，真实HTTP查询状态为{service.get('real_query_smoke_status', '未测')}。这只是本地测试入口更新，不是业务验收或A切换。",
                "",
            ]
        )
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    qa = [
        "# 同题三轮完整问答与辅助核验",
        "",
        "A*=现有业务预分块+共用修正版；B=B1。核验为盲评模型辅助结果，正式人工评审待完成。",
        "",
    ]
    for case_id in by_question:
        first = results["A"][(case_id, 1)]
        qa.extend([f"## {case_id}", "", first["question"], ""])
        if adjudication:
            for item in adjudication["items"]:
                if item["case_id"] == case_id:
                    qa.extend(
                        [
                            "**原件复核提示**：" + item["finding"],
                            "",
                            item["decision"],
                            "",
                        ]
                    )
        for repeat in [1, 2, 3]:
            for arm in ["A", "B"]:
                row, review = (
                    results[arm][(case_id, repeat)],
                    reviews[(arm, case_id, repeat)],
                )
                qa.extend(
                    [
                        f"### {'A*' if arm == 'A' else 'B1'} 第{repeat}轮",
                        "",
                        row["answer"] or "交付失败：" + row.get("error", "未知"),
                        "",
                        f"辅助核验：{'通过' if row['delivered'] and review['full_answer_passed'] else '未通过'}；{review['notes']}",
                        "",
                        "核对原件：" + "；".join(review["source_references"]),
                        "",
                    ]
                )
    (destination / "full-qa.md").write_text("\n".join(qa))
    return {
        "report": str(destination / "report.md"),
        "stats": stats,
        "four_cells": dict(bins),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ["a", "b", "review", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.a, args.b, args.review, args.output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()

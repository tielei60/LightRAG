# 外置证据对照实验：本地运行

本入口不修改 LightRAG 核心、原生上传、查询或 UI，不新建产品主数据，不要求每产品永久分 workspace。`prepare`、`replay` 完全离线；`live` 只在您自己的电脑显式运行。云端开发未使用真实密钥或调用模型。

先按 [本地部署验证](../../docs/LocalValidation.md) 安装依赖并启动原生服务。以下从仓库根目录执行，Python 可替换为本机虚拟环境命令。

## 1. 无密钥验证软件契约

```bash
python -m product.trial_examples.verify_examples
python -m product.evidence_trial prepare \
  --manifest product/trial_examples/synthetic/manifest.json \
  --output data/evidence-trial/synthetic-catalog.json
python -m product.evidence_trial replay \
  --catalog data/evidence-trial/synthetic-catalog.json \
  --case product/trial_examples/synthetic/case_valid_one.json \
  --retrieval product/trial_examples/synthetic/retrieval.json \
  --output data/evidence-trial/synthetic-valid-one.json
python -m product.evidence_trial replay \
  --catalog data/evidence-trial/synthetic-catalog.json \
  --case product/trial_examples/synthetic/case_reject_cartesian_one.json \
  --retrieval product/trial_examples/synthetic/retrieval.json \
  --output data/evidence-trial/synthetic-rejected.json
```

第一份回放应为 `ready`，种子 `synthetic:paired` 补入 `synthetic:common`。第二份应为 `blocked`，不能把允许的 `(计划一,基础)` 和 `(计划二,附加)` 错扩展成 `(计划一,附加)`。这些全部是模拟软件数据，不是真实保险产品或问答质量分数。

CLI 在处理或网络请求前独占保留输出并锁定，每次更新核对同一 run_id；并行相同文件名只能有一个运行成功，不会重复调用模型。拒绝覆盖已有输出，重跑请换一个文件名。`data/` 已在 Git 忽略范围内，真实结果不要提交。

## 2. 验证真实原件定位，而非假称业务通过

```bash
python -m product.evidence_trial prepare \
  --manifest product/trial_examples/jingkangyouhu/manifest.json \
  --output data/evidence-trial/jingkangyouhu-catalog.json
python -m product.evidence_trial replay \
  --catalog data/evidence-trial/jingkangyouhu-catalog.json \
  --case product/trial_examples/jingkangyouhu/case_optional_dependency.json \
  --retrieval product/trial_examples/jingkangyouhu/retrieval_constructed.json \
  --output data/evidence-trial/jingkangyouhu-locator-check.json
```

真实样例只标出了京康优护条款内6个完整原文区间；不覆盖整款产品。保存的 `retrieval_constructed.json` 是构造响应，不是实际 LightRAG 命中记录。真实单位均为 `scope_status: "unreviewed"`，所以该回放应阻断生成。这是在验证定位与未审核保护，不是回避验收。

在用户的实际资料审查后，复制 manifest/case 到 `data/evidence-trial/`：

1. 把 `placeholder:platform-*` 一致替换为现有平台的产品和文件引用，不在这里重新建立产品档案。
2. 确保 `file_path` 与原生查询返回值完全一致，不按 basename 猜测；同名不同文档应有可区分的原生来源标识。
3. 保留原件 SHA 与区间，核实范围组合、共同前提、释义及限制是否足够。仅对真正完成审核的单位改为 `reviewed`；`scope: [{}]` 表示本试验在已选文档内不附加计划过滤，不表示条款没有其他适用条件。
4. 对有计划/选项条件的单位写完整组合，例如 `[{"plan":"计划一","option":"基础"},{"plan":"计划二","option":"附加"}]`；case 的 `scope` 是调用方已选组合。缺少必要维度会阻断，不替用户代选。
5. 使用 schema 2：每份文档必须带审核时的 `extracted_text_sha256`，每个单位必须带审核区间正文的 `text_sha256`。这些是审核记录，不是运行时自动生成的预期值；样例值从原有 `verification.json` 迁入。即使原件字节没有变，解析文本变化一个字符也会失败。旧 schema 1 manifest/catalog 被拒绝。资料替换后重新确认 source_root、原件和解析全文/区间 SHA、边界与依赖，再运行 prepare。不要只改摘要掩盖解析变化。范围审核仍来自原文，不输入独立可编辑的保险结论。

`requires` 只是已核实的必需原文引用，不会自动找全跨条款规则。证据附件保存原文文本、原文哈希和来源身份，能够重建，不是第二份业务真相源。

## 3. 捕获一次真实检索

先在您的原生服务中导入相同原件并检查处理状态。原生导入可能调用模型，请确认本地费用和资料传输后操作。工具不自动导入，也不接管现有索引。

```bash
python -m product.evidence_trial live \
  --allow-network \
  --rag-endpoint http://127.0.0.1:9631 \
  --catalog data/evidence-trial/reviewed-catalog.json \
  --case data/evidence-trial/reviewed-case.json \
  --generate none \
  --output data/evidence-trial/live-context-001.json
```

`/query/data` 不生成最终答案，但关键词阶段仍可能调用模型，所以必须显式加 `--allow-network`。默认 mix、重排关闭、空历史；`--mode naive` 可用于对照。运行目标由 endpoint 决定，不能通过 workspace 请求头动态切产品。

返回数据已经受原生 top-k/token 预算影响。工具后过滤不代表召回前硬过滤；如果正确证据此前已被挤掉，不保证能恢复。可在 case 加 `exact_title_lookup: true`，仅当问题含真实完整标题时做精确补查，结果中单独记录。不会用金标 `expected_evidence_ids` 补答案。

## 4. 显式生成对照答案

本工具不自动加载 `.env`，避免意外使用云端或其他项目密钥。在本地终端按现有环境方式设置：

- `QUERY_LLM_MODEL`：回答模型。
- `QUERY_LLM_BINDING_HOST`，或现有 `LLM_BINDING_HOST`：OpenAI 兼容 `/v1` 地址。
- `QUERY_LLM_BINDING_API_KEY`，或现有 `LLM_BINDING_API_KEY`：只保存在本机环境，不写进命令、case、manifest或Git。
- `QUERY_OPENAI_LLM_EXTRA_BODY`：可选提供方参数 JSON，不能覆盖 model/messages/stream 或放密钥。
- `LIGHTRAG_API_KEY`：若原生服务启用了 API key，在本机环境设置。只支持现有 API-key 身份，不替用户处理登录。

这可能是远程、收费的模型服务；运行 `--generate candidate` 或 `--generate both` 即明确选择向该提供方发送问题及允许文档的原文。不要在未获批准的资料上执行。

```bash
python -m product.evidence_trial live \
  --allow-network --generate both \
  --rag-endpoint http://127.0.0.1:9631 \
  --catalog data/evidence-trial/reviewed-catalog.json \
  --case data/evidence-trial/reviewed-case.json \
  --output data/evidence-trial/live-answer-001.json
```

一次运行只检索一次。基线和候选分别生成一次，共两次回答模型调用，用于对照，不是改写上一份答案。基线仅使用允许文档中的原生检索片段，候选使用补齐后的原文；混合来源图描述不作为事实发送。两者均非原生 `/query` 原始提示的完整复现。默认只捕获，不生成。

字符预算不足、依赖缺失、范围未审时，不调用回答模型。`ready` 只是机械检查通过，不等于结论正确；截断、无效引用和无法获取的阶段会保留在报告中。没有内置模型自评置信度，也不输出无校准的单答可靠百分比。

## 5. 看什么结果

每份 JSON 包含同次运行的：

- `run_id`、时间、源码摘要、解析器版本、目录/manifest摘要、完整case及预算。
- `retrieval_stage` 请求、完整 `retrieval`、耗时；不是第二次查询的替代品。
- `candidate.matches/rejected/blockers`：命中方式、排除原因、依赖/预算问题。
- `candidate.evidence/context`：完整引用原文、文件、页码/区间、范围组合。
- `prompts`、`answers`：实际提示、模型实际请求、完整答案、引用、用量、finish_reason、耗时。
- `unavailable_stages`：原生API未暴露的解析/抽取/合并/关键词模型及预算前候选，明确不可取得，不伪造。这不是整个 LightRAG 内部所有步骤的全量追踪器。

调用前写入 `sending`，中断或异常写入/保留 `error_or_unknown` 或 `sending`，表示结果可能未知；先检查报告与提供方状态，不自动重试。已完成的实验分支不会因后续分支失败丢失。正常结束与可捕获异常会清理当前进程的锁，并保留审计；硬崩溃可能留下 `.<输出文件名>.lock`，不会自动删除或重跑。先确认进程已退出、检查审计和提供方状态，再处理旧锁，避免重复收费。日志不保存配置中的 API keys，错误正文不落盘。

文件权限为600并原子发布，但内容包括问题与原文，可能包含个人信息。默认不上传、不自动清理；您决定保留期限，使用本机文件管理移走或删除旧结果。共享前另行脱敏，只分享必要部分，勿把完整运行目录提交Git。设置 `--output` 到现有敏感文件不是受支持的用法。

对照时分别记录计划串用、金额/比例、关键免责、条件遗漏、可回答覆盖和拒答率，再看延迟/用量。样本通过率不等于每个回答的可靠概率，更不证明全产品、全问题100%准确。

## 6. 开发回归

```bash
python -m pytest product/test_evidence_extract.py product/test_prepare_context.py product/test_manage.py tests/product/ -q
python -m product.trial_examples.verify_examples
python -m ruff check product/evidence_trial.py product/trial_examples/verify_examples.py tests/product/
```

完整设计及未实施边界见 [保险产品优化方案](../../docs/InsuranceEvidenceOptimization-zh.md)。

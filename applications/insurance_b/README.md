# 保险 B 方案应用

B 位于当前 LightRAG 项目的 `applications/insurance_b/`，使用项目现有 SDK、虚拟环境和 Git 仓库。没有创建新项目或 worktree，也没有修改当前 A 的源码、配置、master 数据或9623服务。

当前本地测试候选为 **B1f2**，工作区 `insurance_b1f2`。7产品、21份原件全部导入，86个原生P片段、628实体和625关系，新进程读回通过。95项单元测试、23项集成检查通过；再次导入复用21份资料，新增模型调用0。

A2/B1f2各用29道原题回答3轮，共各87次请求，交付数分别86/83。模型辅助整题通过分别84/81，但已有原件复核争议，不能视为正式业务通过率。A2是现有A业务准备加共用修正版合同的测试副本（A*），不是未修正线上A；B一份输入用了9次选择调用，A预审上限为3，所以这轮不是严格只比较解析/P的实验。当前保留A，B用于本地验证。

详见[最终候选方案](/Users/tielei/workspace/bjlife/LightRAG/applications/insurance_b/FINAL_DESIGN.md)、[完整对照报告](/Users/tielei/workspace/bjlife/LightRAG/applications/insurance_b/output/comparison/final/report.md)和[174份完整问答](/Users/tielei/workspace/bjlife/LightRAG/applications/insurance_b/output/comparison/final/full-qa.md)。B接口为 [http://127.0.0.1:9625/docs](http://127.0.0.1:9625/docs)，入口身份以[服务记录](/Users/tielei/workspace/bjlife/LightRAG/applications/insurance_b/output/comparison/final/service-status.json)及`/health`为准。

当前使用 `_current/`、`profiles_current/`、`snapshots/current.json`，新增保护位于`structure_repair.py`、`input_adapters/`及`query_guard.py`。源码、资料、提示和配置指纹见各运行目录的快照；恢复和核验修正保留了前后证明。下文B0固定链路和首轮诊断是历史记录，不代表当前业务验收状态。

已完成的候选可直接`verify`或`serve`。代码、提示、源文件或关键配置改变后，用新运行号和工作区重新验证；不要把首次准备命令当作覆盖现有冻结批次的更新操作。

## 目录和隔离

```text
LightRAG/
  lightrag/                         # 项目共用 SDK
  .venv/                            # 项目共用运行环境
  applications/
    insurance/                      # 当前 A 实现
    insurance_b/
      cli.py                        # B 入口
      prepare.py                    # 原件、解析与结构检查
      evidence_adapter.py           # 最终 P chunks → 证据合同
      ingest.py / retry.py          # 原生导入、明确重试
      verify.py / server.py         # 新实例读回、B 查询服务
      evaluate.py / fixtures/       # 29 原题、评审与比较
      _frozen/ / profiles/          # 冻结的业务合同与提示
      snapshots/baseline.json       # 共用 SDK 与业务合同指纹
      config/                       # 配置副本及 B 生效配置
      inputs/                       # B 原件副本
      output/                       # B 解析、证据和评估结果
      .runtime/                     # B 数据库数据、日志、迁移备份
      rag_storage/                  # B 的 SDK 本地状态
```

B 使用 PostgreSQL `127.0.0.1:15433` / database `insurance_b`、Neo4j Bolt `17687` / HTTP `17474`，API 端口 `9625`。这两个数据库进程已有独立的 B 数据，迁移时仅移动其目录；没有改用 A 的数据库。所有 B 输入、输出和数据库文件都限制在本应用目录内。A 的代码、配置、数据库和 9623 服务不由 B 启停。

原独立目录 `LightRAG-B` 的剩余安装文件与历史记录收存到 `.runtime/migration-backup/`，不再作为运行入口。

## 复用当前项目配置

按用户要求，项目根 `.env` 已逐字节复制到 `config/project.env`，权限 600。B 的生效配置是 `config/b.env`：模型、接口地址、密钥、温度和 embedding 参数取自该副本；B 的数据库地址、密码、目录和 workspace 保留为自己的值。两个私有文件均被 Git 忽略。

运行时仅显式加载 B 的生效配置，不自动发现根 `.env`。以后修改根 `.env` 不会静默改变已冻结的 B 批次。配置复制记录在 `.runtime/model-config-import.json`，其中不保存密钥值。

当前模型配置：抽取/关键词 `qwen3.8-flash`、回答 `qwen3.7-plus`、embedding `qwen3.7-text-embedding` / 1024 维、rerank `qwen3.7-text-rerank`。温度 0，关闭思考，embedding 显式发送维数，使用 float 编码。

## B0 的固定链路

七产品全部放在同一个 B 工作区，共 21 份原件（14 PDF、7 Markdown），使用同一套 29 道原验收题。PDF 走 Docling，Markdown 走 native，然后明确进入原生 `pending_parse → P → extract → merge → storage` 管线。

P 参数为 2000 token、overlap 100，保留 references。Docling 关闭 OCR、强制 OCR、公式增强及额外多模态分析。编号证据抽取、原文复制、语义校验、合并保护、回答和审核合同仍固定在 `_frozen/`，启动检查其 SHA256。`snapshots/baseline.json` 同时记录当前共用 SDK 的提交和源码哈希；SDK 改变后须明确冻结新批次。

catalog 分别记录实际索引内容、native 提示和编号证据输入。JSON 表格仅在抽取证据输入中按核对后的原单元格序列投影为 Markdown，保留 0、空白、破折号，不改索引内容。证据 offsets 指向 B 投影，物理页/段/表位置取自 sidecar。

## 首次真实导入发现的缓存隔离修复

初始 b0 的 21 份文档技术读回通过，但额外证据审计发现 86 个片段只有 83 份对应的核准选择。三个公司标题/纯图片片段复用了其他文档的原生抽取缓存，导致其他产品实体获得错误来源。缺陷证据保存在 `output/insurance-b/b0/audit/cache-isolation-defect.json`。

b0r1 只在 B 的全局配置适配中，把产品、原件 SHA256、来源路径和进程 epoch 加入原生 extraction cache identity；原生缓存仍保存，以支持 purge/recovery。SDK 和冻结业务合同未改动。读回现在要求每个原生片段的提示都有本文件的核准台账，且文档不得拥有语料中其他产品的实体。新增回归使用真正的 SDK 缓存函数验证相同提示跨产品隔离。

## 从仓库根目录运行

```bash
export EMBEDDING_FUNC_MAX_ASYNC=1
export EMBEDDING_BATCH_NUM=2
export EMBEDDING_TIMEOUT=120
.venv/bin/python -m applications.insurance_b.cli status
.venv/bin/python -m applications.insurance_b.cli storage-start
.venv/bin/python -m applications.insurance_b.cli prepare --parse
.venv/bin/python -m applications.insurance_b.cli ingest
.venv/bin/python -m applications.insurance_b.cli verify
.venv/bin/python -m applications.insurance_b.cli evaluate --repeats 1
```

也可使用 `make -f applications/insurance_b/Makefile status` 等入口，不安装单独的 B 包。先执行一轮 29 题诊断；冻结候选后执行 `evaluate --repeats 3`，得到 87 个请求，独立题数仍为 29。`serve` 在完整读回通过后启动 `http://127.0.0.1:9625`，提供原生查询路由。

查询固定使用 mix、top_k 30、chunk_top_k 12、entity/relation/total token 预算 2000/2500/18000、rerank、空历史及关闭应用答案缓存。配置或代码合同变化后，旧读回报告不能授权新的评估或服务。

写操作使用 B workspace 独占锁。发现任何已有未完成文档时停止导入；检查错误后明确重试：

```bash
.venv/bin/python -m applications.insurance_b.cli retry --document-id <native-document-id>
```

重试使用原生 sticky manual-retry 通道，保留 purge/recovery metadata，要求恰好一个未完成目标。即使只导入部分产品，也先恢复全 B 库的既有合并证据；不隐式调度其他未完成任务。

## Docling 与解析结果

Docling 为 OrbStack 的 `lightrag-docling` 容器，地址 `http://127.0.0.1:5001`，serve 1.38.2，4 CPU、6 GiB、单转换 worker。管理命令为 `/Users/tielei/.config/lightrag-docling/doclingctl`。

21 份原件与已有解析缓存已迁入本目录，不需要重新上传 PDF。当前等待预算约一小时。客户端超时不代表服务端失败；已记录的原任务完成后可以取回结果，再通过缓存生成 sidecar：

```bash
.venv/bin/python -m applications.insurance_b.cli recover-parse --document-id <source-id> --task-id <recorded-task-uuid>
.venv/bin/python -m applications.insurance_b.cli prepare --parse
```

恢复只接受与 B 失败记录匹配的任务 ID，核对原件、返回文件名和缓存 manifest。历史失败报告保留在 `parse-history/`。盈嘉说明书使用过安装阶段已核验的缓存，成本比较必须区分 warm/cold。主机已下载结果会保留；Docling 重启可能主动清空尚未下载的 scratch。

结构机械检查覆盖解析文本哈希、有效解析器、blocks、位置及表引用，不证明阅读顺序、标题语义、跨页表或脚注正确。已记录 `B0-PARSE-FORMULA-001`：盈嘉说明书物理第 2 页的 `1.025` 上标 `N−1` 被展平为 `1.025 N-1`。截图和记录在 `output/insurance-b/b0/visual-review/`；这是业务验收待解决项，B0 未静默改写原文或开启公式增强。

## 评审和 A/B 比较

评估保留请求、答案、原生检索数据、同次生成输入/来源/审核事件、耗时和失败记录。评分依据不进入模型请求。另发诊断请求的上下文不能替代同次输入。未可靠采集到的 token 用量记为未知，不估造费用。

独立核验者根据原件填写 `review-template.json` 的事实、范围、条件/例外、引用、错误和来源位置，模型自审不算业务通过。评分另写 `reviewed-results.json`，保留 raw：

```bash
.venv/bin/python -m applications.insurance_b.cli score --results applications/insurance_b/output/insurance-b/b0r1/evaluation/<id>/results.json --review applications/insurance_b/output/insurance-b/b0r1/evaluation/<id>/review-template.json
.venv/bin/python -m applications.insurance_b.cli compare --a applications/insurance_b/output/baselines/a-reviewed-results.json --b applications/insurance_b/output/insurance-b/b0r1/evaluation/<id>/reviewed-results.json
```

比较只读取放入 B 目录的冻结结果副本，不访问 A 数据库或任务产物。双方须具有一致的题目集、原件集、查询合同哈希，恰好配齐 29 题一轮或三轮。控制条件无法证明相同时仅作诊断，不归因于 P。29 题属于现有小样本，用于调参后不能声称是未见验收集。

输出 A/B 四格和待评审数量，不自动切换入口。技术检查、图谱规模及合成模型结果不构成业务胜出结论。B1 只由真实缺陷触发，使用新的运行 ID/workspace，不覆盖 B0。

## 测试

```bash
PYTHON=.venv/bin/python ./scripts/test.sh applications/insurance_b -q
make -f applications/insurance_b/Makefile check
.venv/bin/python -m applications.insurance_b.cli smoke
```

smoke 使用真实 B 数据库和独立 `insurance_b_smoke_*` namespace，替代模型、embedding、rerank，外部模型 API 调用为 0。覆盖 P、图/向量、自然回答/审核、HTTP、新进程读回、重复导入、失败/明确重试和删除清理。真实模型导入与 29 题业务成绩单独报告。

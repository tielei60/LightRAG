# master 数据库工作区

日常服务和新增产品统一使用 `WORKSPACE=master`。只有明确需要隔离实验时，才指定另一个工作区。导入批次是一次操作的记录，不是新的工作区。

保险业务代码和命令统一使用 `applications.insurance`，原 `product/` 目录和转发入口已删除。服务通过 `applications.insurance.server` 启动，当前操作命令以本文件中的新入口为准。

## 当前本机配置

| 项目 | 配置 |
| --- | --- |
| Python / WebUI | `http://127.0.0.1:9623/workspace/`，`make -f applications/insurance/Makefile start` |
| 管理页面 | `http://127.0.0.1:9623/webui/` |
| 默认服务配置 | `output/master/runtime.env`，根 `.env` 使用相同数据库配置 |
| Neo4j | 现有 Homebrew 2026.09.0，Bolt `127.0.0.1:7687`，database `neo4j` |
| PostgreSQL | Homebrew `postgresql@17` 17.11，`127.0.0.1:5433`，database `lightrag_products` |
| pgvector | Homebrew 0.8.6，SQL extension `vector` |
| 数据工作区 | PostgreSQL `workspace='master'`，Neo4j `master` 标签 |

原有 PostgreSQL 14 继续使用5432，未替换或升级其数据。Neo4j 密码已按用户授权重置，认证已恢复，连接仅监听本机。密码仅写入被Git忽略的私有配置，权限600，不放入本示例或文档。配置字段示例见 [database.env.example](database.env.example)。

数据库通过 Homebrew 服务管理：

```sh
brew services start postgresql@17
brew services start neo4j
make -f applications/insurance/Makefile status
make -f applications/insurance/Makefile start
make -f applications/insurance/Makefile stop
```

## 数据归属

| 后端 | 原生实现 | 数据 |
| --- | --- | --- |
| Neo4j | `Neo4JStorage` | 实体、关系、原文描述、类型、来源与其他图属性 |
| PostgreSQL | `PGKVStorage` | 正文、片段、抽取及摘要缓存、文档实体/关系恢复锚点、实体/关系来源追踪 |
| PostgreSQL | `PGVectorStorage` | 片段、实体、关系的1024维向量及其元数据；HNSW索引 |
| PostgreSQL | `PGDocStatusStorage` | 处理状态、错误、片段归属、`kg_write_state` / purge journal 等恢复metadata |
| 资料目录 | `docs/`、`output/` | 原件、解析与分段产物、证据选择、导入批次、冻结提示、验收及备份产物 |

不需要 Apache AGE。`master` 是应用工作区，不是 Neo4j database 名。不要用 `POSTGRES_WORKSPACE` 或 `NEO4J_WORKSPACE` 把某一个后端指到其他工作区。

`rag_storage/master/` 可能由原生初始化保留为空的运行目录；它不再是正文、图谱或向量的运行存储。`database-snapshots/` 中JSON、GraphML是从数据库导出的核验/备份产物，运行查询不读取这些文件。

## 本次重导

初始数据库批次为 `database-master-initial`。它重新核对三批既有来源的全部412个已核抽取选择，通过原生SDK回放写入数据库；原件和分段不改写，未重新调用抽取模型，向量通过配置的embedding重新写入。

```sh
.venv/bin/python -m applications.insurance.evidence_import --batch database-master-initial --approved-batches two-products-v5 merge-four-products incremental-product-5 --apply
```

上面仅记录已完成操作；所列旧来源批次已清理，不能作为当前复现命令，也不能对已有数据库再次运行。新导入默认会检查数据库全部命名空间是否为空；已有数据时要求显式增量。

累计登记为 `output/workspaces/master/registry.json`，包含后端、数据库地址、向量模型/维度和已完成批次。初次重导和后续追加均在全库证据、图谱、向量及原数据保留校验通过后登记。

旧文件库及旧顶层测试批次已清理。初始批次已将必要准备产物复制进自身 `sources/`，当前master的运行、核验和追加依赖登记的 `database-master-initial`、`master-product-6`、`master-product-7`，不递归读取旧来源批次。默认9623服务使用数据库 `master`。

## 后续新增产品

每次使用新批次名，工作区保持默认 `master`。先准备和预检，再停Python服务执行同库追加：

```sh
.venv/bin/python -m applications.insurance.prepare_batch --batch NEW_BATCH --products 产品目录 --config output/master/runtime.env --append
make -f applications/insurance/Makefile BATCH=NEW_BATCH append-preview
make -f applications/insurance/Makefile stop
make -f applications/insurance/Makefile BATCH=NEW_BATCH append-import
make -f applications/insurance/Makefile verify
make -f applications/insurance/Makefile start
```

准备器不会以本地存储目录是否存在判断数据库工作区。导入器实际查询数据库并核对累计登记；向量模型/维度和后端必须匹配，重名产品、规范化文件名及重复内容会拒绝写入。更新已有产品属于单独的替换流程，不能按新增产品重复导入。

同库Python服务和SDK导入不能同时写。启动与导入使用同一个工作区互斥锁；数据库增量保存写前数据库快照及操作journal，失败保留原始证据供检查，不自动清库或删除回执重放。准备后仍需冻结本批使用的抽取提示，并让该批 `PROMPT_DIR` 指向副本。

新包内部导入更新会改变抽取器源码合同哈希，新批次按当前源码计算合同。已完成批次的合同、台账、回执和登记记录保持原样，读回仍使用各批冻结配置与证据重新核验。当前master没有中断批次；旧源码合同的部分台账会被新代码拒绝直接续跑，相同当前合同的部分台账仍可恢复。复用已核选择时使用现有 `--approved-from` 或 `--approved-batches` 重新校验，不修改旧哈希或取消校验。

## 备份和以后上线迁移

初次五产品数据库迁移的停写窗口生成了 `output/database-migration/native-backup/`：PostgreSQL custom dump、Neo4j原生dump和不含凭据的产品资料归档。摘要及恢复验证保留在该目录同级的 `native-backup-manifest.json`、`cutover-plan.json`、`postgres-restore-test.json`。这些原生dump对应前5款15文档/412片段时点，不包含后续第6、7款追加；它们不是当前7款master的最新备份。导出的主工作区快照可用于逐行核对，但不能替代原生dump；它不包含所有数据库级索引、扩展和管理对象。

PostgreSQL 使用与服务器匹配的17版工具，备份/恢复示例：

```sh
pg_dump --host PG_HOST --port PG_PORT --username PG_USER --dbname lightrag_products --format=custom --no-owner --no-acl --file lightrag_products.dump
pg_restore --host NEW_HOST --port NEW_PORT --username NEW_USER --dbname NEW_DATABASE --exit-on-error --no-owner --no-acl lightrag_products.dump
```

通过权限600的 `.pgpass` 或部署环境注入凭据，不将密码写入命令行。目标 PostgreSQL 需安装兼容的 pgvector，数据库角色需具备表和扩展所需权限；资料、向量模型和维度保持一致。

Neo4j原生dump要求停止数据库；备份两种数据库时应使用同一个应用停写窗口：

```sh
brew services stop neo4j
neo4j-admin database dump neo4j --to-path=/absolute/backup/directory
brew services start neo4j
# 在停写的、版本兼容的目标Neo4j上执行：
neo4j-admin database load neo4j --from-path=/absolute/backup/directory
```

本机 `neo4j` database 还包含原有非master数据，因此原生 `neo4j.dump` 是整个database的备份。应用工作区核验导出只包含 `master`，迁移时应根据目标是否接收整个database选择恢复方式，不能覆盖有其他业务数据的目标库。

同时搬迁原件、解析产物、冻结提示、已核台账和workspace registry，保持资料相对目录结构。数据库密码在目标环境重新配置；部署目录变化时校正资料和提示路径。目标主机/库名改变后，应先完成全库核验，再更新registry中的数据库地址合同，不能把原地址不一致检查直接关掉。根 `.env` 与 `output/master/runtime.env` 必须指向同一套目标数据库。

上线恢复的验收以最新累计registry及读回报告为准，核对文档、片段、实体、关系及各向量ID/维度，缓存、来源追踪和恢复锚点的完整性，以及实际检索问答。初次迁移时的15文档、412片段、497实体、646关系只是当时基线。存储核验通过不等于业务问答全部正确。

## 后续增量记录

`master-product-6` 已向相同数据库master追加京福宏裕（尊享版）终身寿险（分红型），新增3文档82片段，当时累计6产品、18文档、494片段、597实体、754关系，全库及原有5产品的数据保留校验通过。

`master-product-7` 通过完全切分后的 `applications.insurance` 入口追加京福盈嘉终身寿险，新增3文档69片段，当前累计7产品、21文档、563片段、680实体、843关系。独立只读SQL/Cypher核验47项通过，旧18文档、494片段、恢复锚点逐行一致，旧561缓存完整保留，累计640缓存（563抽取、77摘要）；原有非master的30个Neo4j节点保留，跨工作区边为0。服务仍为9623的master，模型和预算沿用，控制配置只更新了冻结提示与回答日志路径。

两次追加的数据库写前快照、写后导出和journal分别保存在对应批次目录；第7款独立核验为 `output/master-product-7/independent-database-acceptance.json`。最新累计来源以 `output/workspaces/master/registry.json` 为准。

第7款技术流程和原数据保留通过，业务验收另行记录：8题均交付且主问核心正确，仍有1题额外跨产品断言错误、2题精度备注；图审确认2组退保材料误归理赔材料。问答全文和图谱业务验收未完全通过，详见该批 `business-review.json`、`extraction-quality-review.json`，不能以数据库核验通过替代业务结论。

## 已核分类更正

第7款的两组退保/解除合同材料已通过 `classification_repair.py` 修为“保全材料／需要保全材料”，未重导产品或改写历史台账。修复前停止同库服务、取得工作区写锁并完整备份；新配置重新校验证据后，保存已核差异和journal，再同步修正两条原抽取缓存、两个图节点、两条关系及两条关系向量。只改类型和关系关键词，保留名称、原文、来源、权重及向量 `file_path`；实体向量保持不变。不能仅改图后保留错误的抽取缓存，否则来源重建可能恢复旧分类。

更正记录及写前/写后快照位于 `output/master-product-7-fixes/`，journal为 `classification-repair-journal.json`，独立数据库核验为 `independent-database-acceptance.json`。56项独立只读检查通过：master仍为21文档、563片段、680实体、843关系、640缓存；仅允许的两条缓存和两条关系向量行变化，图中仅两个类型和两个关键词变化。文档、片段、恢复锚点、来源追踪及其他向量不变，15个历史清单、台账和冻结提示摘要不变。其他工作区数据保留。

这是有明确原件证据和旧值前提的分类更正入口，不是任意图编辑或自动回滚工具；多来源目标会拒绝部分修正，部分写入失败须保留journal及备份检查，不能盲目重放。上线迁移时应同时保留更正记录及当前数据库，不能用旧已核输出覆盖更正后的缓存和分类。

上述初次迁移的原生dump属于5产品时点，不会因后续追加自动变成最新备份。上线迁移前应在新的停写窗口重新生成成套原生备份，并按最新累计清单核验数量和来源。

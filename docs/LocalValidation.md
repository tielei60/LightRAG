# 云端开发，本地部署验证 LightRAG

本页只准备当前 LightRAG 分支的运行环境，不改变产品抽取、合并或检索方案。云端只保留源码、空凭据模板和虚构样例；真实模型地址和 API 密钥只由你在本地填写。不要把密钥发到聊天、提交到 GitHub、放进命令行参数或复制进问题截图。

本流程使用原生 LightRAG API、WebUI、JSON / NetworkX / NanoVectorDB 本地存储，不需要另装问答平台、Docker、PostgreSQL 或 Neo4j。以下命令针对 macOS / Linux。

## 1. 使用全新目录

不要直接切换正在运行旧知识库的仓库，也不要复制旧的 `.env`、`rag_storage`、`inputs` 或上传回执。

```sh
git clone --single-branch --branch codex/lightrag-cloud-setup \
  https://github.com/tielei60/LightRAG.git lightrag-local-validation
cd lightrag-local-validation
```

如果这个新验证目录已经创建，先停止其中的服务，再使用 `git pull --ff-only`。本次新增工具不会修改仓库根目录的 `.env`，也不会导入、清空或迁移已有知识库。

## 2. 安装锁定依赖并构建界面

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 [Bun](https://bun.sh/docs/installation)。本次云端构建使用 Python 3.12、Bun 1.3.14。缺少工具时按官方文档安装，不需要把 API 密钥配置到云端或 GitHub Actions。

```sh
uv sync --frozen --python 3.12 --extra api --extra pytest --no-dev
(cd lightrag_webui && bun install --frozen-lockfile && bun run build)
.venv/bin/lightrag-download-cache --tiktoken --cache-dir .cache/tiktoken
```

这一步会从包仓库及官方 tokenizer 缓存地址下载文件，不调用大模型。没有安装额外数据库，也不把 Ollama 模型运行在本机。若本机确实使用 SOCKS 网络代理，按需要补充 `.venv` 中的代理支持：

```sh
uv pip install --python .venv/bin/python 'httpx[socks]==0.28.1' 'requests[socks]'
```

普通 Markdown 和 PDF 文字解析不需要 spaCy 语言模型、MinerU 或 Docling。DOCX 的 `smart_heading`、复杂 OCR 等属于另外的可选配置，不能把基础环境通过当成这些能力也已验证。

## 3. 创建私有本地配置

```sh
.venv/bin/python scripts/local_validation.py init
.venv/bin/python scripts/local_validation.py check
```

`init` 只创建 `data/local-validation/.env`，权限为 600；文件存在时拒绝覆盖。该目录已被仓库的 `data/` 忽略规则覆盖。第二条检查依赖、已构建页面和存储隔离，不发送任何模型请求；空模型配置会明确提示尚未完成。

用本地代码编辑器打开 `data/local-validation/.env`，填写以下字段，**不要把内容粘贴到终端命令或聊天中**：

- `LLM_BINDING_HOST`：真实 OpenAI 兼容接口的 base URL，通常包含 `/v1`。
- `LLM_BINDING_API_KEY`、`LLM_MODEL`：本地保存的密钥和该接口实际支持的模型名。
- `EMBEDDING_BINDING_HOST`、`EMBEDDING_BINDING_API_KEY`、`EMBEDDING_MODEL`：实际向量服务信息。
- `EMBEDDING_DIM`：该向量模型的真实输出维数。模板中的 1024 只是待确认值，不是所有模型的通用值。

可以在本地查看现有私有配置并手工填写必要模型字段，但不要复制旧存储路径或整个文件。模板刻意不填模型名和服务商地址，以免把文档中的示例误当成真实可用配置。

其他隔离字段保持不变：端口 **9631**、工作区 **local_validation**、存储与输入目录都在 `data/local-validation/` 内。它不会占用旧的 9621–9623 端口。工具会拒绝将路径改到旧知识库、数据树中的符号链接重定向、外网监听、额外模型角色或 Langfuse 设置。启动只向子进程保留必要的系统、语言、代理和证书环境变量，应用设置只来自这份专用配置，不继承旧服务的认证、SSL 或日志设置。

请使用没有旧 LightRAG 环境变量的终端。检查发现冲突时只列出变量名，不显示值；不要通过把整个 shell 环境贴出来排查。

```sh
.venv/bin/python scripts/local_validation.py check --require-models
```

此检查只验证结构和必填项，不能证明 API key 有效、模型存在或服务有额度。

## 4. 本地启动和健康检查

```sh
.venv/bin/python scripts/local_validation.py start
```

保持这个终端打开。在另一个终端进入同一仓库执行：

```sh
.venv/bin/python scripts/local_validation.py verify
```

验证工具只读取 `127.0.0.1:9631`，检查健康、目标工作区、数据路径和页面。打开：

- 管理界面：http://127.0.0.1:9631/webui/
- 原生独立查询界面：http://127.0.0.1:9631/workspace/
- API 文档：http://127.0.0.1:9631/docs

这些是你本机的地址，不是云端网站。默认仅监听回环地址、使用本地匿名模式，不应直接改成 `0.0.0.0` 对外开放。要结束服务，在启动终端按 Ctrl+C；以后可以使用同一条 `start` 命令再次启动。这里没有设置自动开机服务或公网发布。

## 5. 先用虚构样例验证真实链路

此步骤由你本地执行，**会向你配置的模型服务发送样例内容，并可能产生 API 费用**。使用仓库中的 `examples/local_validation/smoke.md`，不要先上传真实客户或保险业务资料。

1. 在 9631 管理界面上传该文件一次。等待状态从受理变为处理完成，查看是否有解析、抽取或 embedding 错误。上传受理不等于成功。
2. 查看文档来源片段和图谱，再用同一工作区提问：
   - “普通图书可以借几天？什么条件下可以续借？”应包含14天、未逾期、无人预约、仅一次、增加7天。
   - “参考资料也可以外借14天吗？”应明确不能外借，不能套用普通图书规则。
   - “周一开放吗？”应回答不开放。
3. 检查答案引用是否确实指向样例文件，答案是否受原文支持。不要把“有答案”当成正确。
4. 停止后重启，再核对文档状态并提问，确认本地持久化仍在。不要重复上传同一文件来代替重启检查。

只有这几步在你的真实模型配置下通过，才能说真实导入→解析→抽取→向量/图存储→查询链路跑通。此样例不能代替保险业务验收。失败时先检查脱敏错误、endpoint、模型名和向量维数，不要删旧库或盲目重导。已有向量数据后切换 embedding 模型，需要另建验证实例，不能混用向量空间。

## 6. 不使用密钥的离线检查

先完成上面的依赖和 tokenizer 下载。以下测试不访问模型服务、不导入真实资料。`PYTHON_DOTENV_DISABLED=1` 防止测试自动加载本地 `.env`；仍应在没有真实模型变量的终端运行。

```sh
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest \
  tests/setup/test_local_validation.py \
  tests/kg/json_impl tests/kg/nano_impl tests/kg/networkx_impl \
  tests/parser/test_markdown.py \
  tests/parser/markdown/test_parser.py tests/parser/markdown/test_ir_builder.py -q
```

云执行环境禁止多进程 Unix socket，因此云端对两项 `real_manager_persists` 测试的验证受限；本地 macOS/Linux 应先正常运行全部上述测试，不要默认跳过。如果确实受到同类限制，记录原因后才使用 `-k 'not real_manager_persists'`。云端没有执行真实模型链路，也没有验证你的本地模型凭据。

## 7. 提交和分享前

```sh
git check-ignore data/local-validation/.env data/local-validation/rag_storage/example.json
git status --short
git diff --cached --stat
```

第一条应显示两个文件都被忽略。不要使用 `git add -f` 提交运行目录。代码提交只应包含源码、测试、文档和空模板；日志、缓存、输入文档、模型输出、`.env` 均留在本地。Git 忽略只是防误提交，不是加密或完整密钥扫描；分享日志前仍需检查并删去凭据、认证头、私有端点及文档内容。

若密钥曾误提交，仅删除文件或新增忽略规则不够，应在服务商处撤销并更换密钥，再处理历史。当前准备包不要求你向云电脑提供真实密钥。

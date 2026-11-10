# 产品资料首批导入

独立仓库基于官方 `v1.5.7`，提交 `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`。

- `origin`：用户 fork `tielei60/LightRAG`。
- `upstream`：官方 `HKUDS/LightRAG`。
- `1.5.7`：保留官方基线。
- `codex/product-import-1.5.7`：产品配置、提示资源和操作支持。

本目录不修改 LightRAG 核心、解析器、抽取协议、检索或回答流程。使用官方 API 服务与原生 WebUI，首次只导入三款产品，由用户检查原文、图谱和问答。旧项目仅提供模型连接配置，不读取旧工作区。

## 本机入口

- 文档、图谱和检索：[http://127.0.0.1:9621/webui/](http://127.0.0.1:9621/webui/)
- 独立提问：[http://127.0.0.1:9621/workspace/](http://127.0.0.1:9621/workspace/)

服务只监听本机回环地址，使用官方本地匿名模式；没有新建账号。私有模型凭据保存在根目录 `.env`，权限为600，不进入Git。

## 首批范围

| 产品 | 本批资料 | 检查重点 |
|---|---|---|
| 京安泰 | 条款、产品说明书、投保及核保规则 | 多计划、责任选择、职业和风险保额条件 |
| 京康优护 | 条款、产品说明书、投保及核保规则 | 计划一/二、基本/可选责任、免赔与等待期 |
| 京富盈利（七周年） | 条款、产品说明书、投核保规则 | 生存/满期/身故给付及投核保条件 |

共9份文字资料，原件在同级平台项目的 `docs/insurance_docs` 下只读取得。费率和现金价值表未导入，本批不提供标准金额查算。原文中的比例、年龄和期间等数字保留。

原件清单和SHA256位于 `output/first3/manifest.json`，上传受理回执位于 `output/first3/uploads.json`；最终逐文档读回结果位于 `output/first3/final-document-status.json`，技术核验报告位于 `output/first3/technical-verification.json`；实际文档结果以原生 `/documents/track_status/{track_id}` 和工作区状态为准。受理成功不等于处理完成，更不等于业务质量通过。

## 配置与扩展

本批通过官方配置接入模型和提示，不引入平台业务依赖：

| 项目 | 本批选择 |
|---|---|
| LLM绑定 | 官方OpenAI兼容绑定 |
| 抽取、关键词、回答模型 | 明确配置为 `qwen3.8-flash`，关闭思考，temperature为0 |
| 向量模型 | `qwen3.7-text-embedding-flash`，1024维 |
| 存储 | 官方JSON、NetworkX、NanoVectorDB本地组合 |
| 新工作区 | `products_first3`，位于 `rag_storage/products_first3` |
| PDF | 官方legacy解析及R递归分块 |
| Markdown | 官方native解析及P分块 |
| 普通分块参数 | 1200 token、100 overlap；P采用官方独立默认配置 |
| 抽取语言和协议 | 中文、官方JSON抽取协议 |
| 抽取轮次 | `MAX_GLEANING=0`，即初次抽取后无额外gleaning |
| 单次抽取上限 | `MAX_EXTRACTION_ENTITIES=200`、`MAX_EXTRACTION_RECORDS=500`；官方默认是40/100 |
| 重排 | 本批未配置重排服务，保留原生检索，页面参数按实际能力使用 |

数量上限是每个片段的一次模型响应，不是每份文件或每个产品总量；调宽不等于不限量，也不要求凑满。图谱的描述合并等原生调用仍可能发生，“一次抽取”不代表整个导入仅调用一次模型。

自定义实体类型和关系指导位于 `product/prompt_profiles/entity_type/products.yml`，通过官方 `PROMPT_DIR` / `ENTITY_TYPE_PROMPT_FILE` 加载。优先保留产品归属、条件、否定、例外和原文支持，不拆出大量无用细事实。输出示例只描述官方结构，不添加虚构业务案例。

## 运行

在仓库根目录执行：

```sh
make -f product/Makefile start
make -f product/Makefile status
make -f product/Makefile preview
make -f product/Makefile import
make -f product/Makefile stop
```

`import`要求已存在准确的本机清单，核原件摘要、三个产品、官方版本及新目标后调用官方上传接口。已有上传回执时拒绝自动重放；中断或未知先核原请求，不删除回执来盲目再导。日志位于 `output/server.log`，进程号位于 `output/first3/server.pid`。

环境使用Python 3.12，安装依赖见 `requirements.lock`，再以 `uv pip install --no-deps -e .` 安装本仓库。原生WebUI静态制品取自同版官方wheel，放在Git忽略的 `lightrag/api/webui`；可以依官方说明重新构建。`.env`、inputs、rag_storage、output和真实原件均不提交到Git。

## 用户检查

先核文档状态，再查看实体/关系和来源片段，最后用已有问题提问，关注计划差异、责任选择前提、免责与例外是否正确。导入技术结果与用户业务验收分别记录；用户检查前不扩展至其他产品、不自动批量评分或反复调参。本机浏览器自动化当前不可用，页面可访问性由HTTP检查，实际交互由用户核验。

## 结构与效果检查边界

PDF的legacy解析保存展平文字，不提供逐页结构sidecar。与同一解析器输出一致只能证明提取结果已保存，不能证明表格行列、注释和全部图文完整。官方P分块依赖解析sidecar；没有sidecar会退为R，不能仅改一个分块参数就恢复PDF结构。Markdown的原生标题和表格结构也不能自动证明跨条款条件已完整。

检查按“原件→解析→片段→实体关系→实际上下文→原答”定位。先核原文是否读全、条件是否保留及产品归属，再判断应调整解析、分块、唯一抽取profile还是查询配置。实体多或上下文短不作为质量目标。当前交付用户检查后，再根据具体问题确定必要候选；改配置不会更新已生成图谱，也不应删除回执盲目重导。

原生 `/query/data` 可查看实体、关系、片段及引用，不生成最终答案，但关键词提取等仍可能调用模型。分别请求data和answer会独立执行，不能把前者自动当作后者实际使用的上下文。源码中的产品专属业务逻辑保持零新增。

## 京康优护上下文优化候选

独立候选入口：[http://127.0.0.1:9622/webui/](http://127.0.0.1:9622/webui/)，提问入口为同端口 `/workspace/`；工作区 `jingkangyouhu_context`。原9621首批工作区继续保留。本候选仅处理京康优护的条款、说明书和投核保规则，问题由用户自行提出和验证。

```sh
make -f product/Makefile BATCH=jingkangyouhu-context status
make -f product/Makefile BATCH=jingkangyouhu-context stop
make -f product/Makefile BATCH=jingkangyouhu-context start
```

`output/jingkangyouhu-context/runtime.env` 是候选私有配置，启动时只传给该进程。清单、逐份受理回执、准备后的请求、原文范围映射和检查记录均在同一忽略目录。已有回执禁止盲目重导。

`prepare_context.py` 接受经过检查的原文边界，读取PDF原文或Markdown，在每个片段添加经原件核实的产品名称、文件名和物理页范围，原文正文不改写。边界清单与摘要验证后，使用官方 `/documents/text` 的 `fixed_token`、`split_by_character_only` 保持片段，不替换内部函数；超过声明的8192 token上限会明确失败，不静默再切开。本候选最大片段低于2500 token；上限是接口的输入兼容参数，不是调用预算。

条款按已检查的完整条款边界准备，保险责任2.5含基本/可选责任及年龄、选择前提，职业或工种变更6.4保留完整通知义务和三个分支。释义整体保留，说明书与投核保规则各整体保留，因此不会因进一步切分丢掉其计划对应表。此安排适用于本批实际长度，未声称自动适用所有产品。边界及产品身份作为本机资料清单，不写进业务代码。

抽取仍使用唯一 `entity_type/products.yml`，增加Rule类型及产品限定命名、计划—责任—规则关系指导；共享责任可共用，不同适用条件须分清，不能为了连通而虚构关系。原生回答附加指导位于 `user_prompt/products.md`，通过 `USER_PROMPT_PREFIX_FILE` 加载，要求计划未明确且有差异时区分说明，不能默认选择或混用规则。用户仍需检查实际回答是否遵循。

候选是“输入上下文＋条款边界＋提示指导”的组合改进，不将效果单独归因于提示。没有迁移旧图/缓存、手工补边或修改LightRAG核心。原生来源文件名指向原件身份，但文本导入不会上传原件下载副本；完整原件仍在原只读目录，PDF页码指物理页而非页脚印刷页码。原生图和来源可用于检查，不能把连接数增加当作语义验收通过。

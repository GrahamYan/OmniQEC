# OmniQEC 当前实现架构与数据流（as-built，2026-09-02）

> 当前协议 profile：**`telescoping_structural_v3_6`**（2026-09-03，decoder = `telescoping_cpu`，见「追加（八）」；本段其余是 v3.5 的描述，保留为历史）。v3.5 = **`bposd_structural_v3_5`**（见 §6 与 §10 末节「2026-09-02 追加（六）」与「追加（七）」；v3.4 的执行几何见「追加（五）」）。噪声模型 = 论文 SI C 的 **`uniform_paper_v1`**（单一 p=0.005 的均匀电路级去极化，**无 idle 噪声**），SEC = **`twobasis_nonint_v1`**（两基每轮都测、**不交错**、基内层序 K=16 候选 + 每候选每基 400 次 ISD 认证；代码默认、strict 拒绝其它；LRC 保留为历史策略 `lrc_ranked_v1`，见「追加（七）」）；其余沿 v3.4：BPOSD 单解码器、4-seed `d_search`、(k, d_search) 结构保留通道、3S+1R slow 槽、8S:4R 父代、固定 T=12 终局审计、racing 关闭；search 每基固定 16 个 shot shard、final 每基 128 个，decode-shard watchdog = max(1800 s, 8 s/shot × shard shots)，watchdog 超时不重放、瞬态错误（worker 被杀 / DEM 丢失）同请求重放一次，完整性门防止难码被超时静默筛掉；只因超时而不完整的候选以 `decode_infeasible` 终态退出（不入排名/终局、不重排队），不再停止整个点。

- 状态：**实现快照**，非设计稿。01/02/03/03a 描述*应该建成什么*；本文描述*现在运行的是什么*。
- 用途：新人入场地图；改动前的影响面判断；论文 Methods 的事实依据。
- 冲突时的权威顺序：**代码 > 本文 > 设计稿**。设计稿定稿后未实施的条目在 §11 单列。
- 规模：`omniqec_agentic/` 约 27k 行 Python（其中 tests/ 4.2k），363+ 个测试（2026-09-02 v3.5 之前 `pytest -q` 实测 363 passed：300 基线 + v3.4 的 `test_v34_decode_budget.py` 与 `test_hpc_v34_scripts.py` 等；v3.5 再加 `test_v35_noise_profile.py` 与 `test_hpc_v35_scripts.py`，以实际输出为准）；同日本机 16/32-shard 端到端 canary（含强制 decode_infeasible 场景）通过。

---

## 1. 分层与依赖规则

```text
                      ┌─────────────────────────────────────────┐
  workflow/           │  LangGraph 图：intake → [turn]* →        │  ← 唯一 import langgraph 的包
  （框架层）           │  final_slow → report + SqliteSaver       │
                      └──────────────────┬──────────────────────┘
                                         │ Policy 协议（依赖注入）
                      ┌──────────────────┴──────────────────────┐
  orchestrator/       │  AgenticPolicy（LLM 工具循环）            │  policy 层
  tools/              │  FallbackPolicy（确定性降级）             │  ——决定"测谁、花多少、往哪探"
                      │  tools/：read / draft / policy 三档权限   │
                      └──────────────────┬──────────────────────┘
                                         │ TurnPlan（schema + 引用校验）
                      ┌──────────────────┴──────────────────────┐
  funsearch.py        │  Generator：archive → prompt → 程序 →     │  fast 环
  construction/       │  sandbox → 结构检查 → MAP-Elites archive  │
  memory/archive.py   │                                          │
                      └──────────────────┬──────────────────────┘
                                         │ promotion_queue.jsonl
                      ┌──────────────────┴──────────────────────┐
  evaluation/         │  build → SEC 综合 → DEM → 解码 → 证据     │  slow 环
  sec/ verification/  │  子进程池 + 硬超时 + anchor-p racing      │  ——protocol 层，agent 不可触碰
                      └──────────────────┬──────────────────────┘
                                         │ slow_results / rankings
                      ┌──────────────────┴──────────────────────┐
  memory/             │  证据行 → 回测器 → ledger → bandit        │  学习层
  orchestrator/manual │  指纹 / 已知码 / 覆盖度 / 技能手册         │
                      └─────────────────────────────────────────┘
```

**硬性依赖规则**（`workflow/__init__.py` 明写）：只有 `workflow/` 依赖 langgraph。换框架 = 重写这一薄层，`orchestrator/`、`evaluation/`、`sec/`、`construction/`、`verification/` 全部不动。pydantic 允许全层使用。

**两层分权**是整个系统的核心设计约束：

| 层 | 谁决定 | 例子 |
|---|---|---|
| **policy** | LLM agent | 提多少程序、给哪个 family、晋级谁、探索温度、假设提案 |
| **protocol** | 冻结的确定性模块 | 噪声点、shot 数、解码器、SEC 策略、CI 算法、假设的支持/证伪判定 |

工具层用 scope 强制这条线：`read` 纯查询，`draft` 沙箱试算，`policy` 仅限 ledger 提案与交计划。**agent 说的话永远不会变成记录**——返回值全部是机器算出来的事实。

---

## 2. 两个 LLM 角色

系统只有两个 LLM 角色，不是多 agent 编排：

| 角色 | 位置 | 输入 | 输出 | 每轮调用 |
|---|---|---|---|---|
| **Policy（Orchestrator）** | `orchestrator/loop.py` | 10 个带来源标注的上下文块 + 工具返回 | 一个 `TurnPlan` JSON | 1 次计划 + ≤ tool_budget 次工具 + 结构化修复（预算 4，每成功一轮工具回补 1、上限 4）；每轮 LLM 调用总数 ≤ tool_budget + 2 |
| **Generator** | `funsearch.py` | archive parents + 策略简报 + family 配额 + slow 反馈 | N 段 `construct(rng)` Python | 每 generation 1 次（+ 续写/补齐重试） |

两者共用 `llm/` 下的 completion-only 客户端（deepseek / openai_compat / anthropic / openrouter），**不使用 langchain 的模型集成**。

Policy 的循环协议是"每条消息恰好一个 JSON 对象"：`{"tool": name, "arguments": {...}}` 调工具，`{"submit_plan": {...}}` 收尾。原始响应**先落盘再解析**——解析失败时证据不会丢。

---

## 3. 一个 turn 的完整数据流

```text
 ┌── orchestrator_turn ────────────────────────────────────────────────┐
 │ 1. 装配 S-1..S-10 上下文块（每块带 origin + sha256:16）               │
 │ 2. LLM 工具循环：query_archive / query_coverage / query_evidence /   │
 │    fingerprint_lookup / known_code_lookup / acquisition_score /     │
 │    precheck_draft / ledger_propose / ledger_retract / ...           │
 │ 3. submit_plan → validate_turn_plan()                               │
 │    · schema（pydantic，extra=forbid）                                │
 │    · 引用校验：cites 只能是本轮 S-*、已知 code_hash 前缀、H-*/I-*      │
 │    · family_mix 归一、program_count ≤ cap、promotions ≤ promote_cap  │
 │    · 隔离候选不可晋级、偏离 acquisition 基线须给带引用的理由            │
 │ 4. 失败 → 结构化修复（预算 4，每成功一轮工具回补 1）→ 耗尽 →             │
 │    FallbackPolicy（严格模式下直接抛错）                                 │
 │ 落盘：turns/turn_NNNN/plan.json + orchestrator/{llm_calls.jsonl,     │
 │       llm_response_NN.txt, tool_transcript.json, turn_plan.json,     │
 │       context_blocks.json, events.log}                               │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │ plan
 ┌── execute_plan ──────────┴──────────────────────────────────────────┐
 │ family_mix ──最大余数法──→ family_quota（整数槽，合计 = program_count）│
 │ family_mix + strategy_mix + 已裁定 insight（≤12 条）──→             │
 │   strategy_brief（v3.3：directives / 反思 advice / 原始 generation   │
 │   notes 不再逐字进 Generator，只留给 Policy 与审计）                   │
 │ 上一轮 slow 反馈 overlay ──→ prompt_context_slow_aware.json          │
 │                                                                     │
 │  ┌ fast 环（funsearch）────────────────────────────────────────┐     │
 │  │ 父代 = 机器配额：跨 turn 8 structural + 4 runtime，按        │     │
 │  │   S,S,R,… 2:1 交错（渲染窗口前 6 条 = 4S+2R），覆盖 overlay   │     │
 │  │ → prompt → LLM → 提取代码块 → sandbox 执行 construct(rng)    │     │
 │  │ → construction/ 编译成 HX/HZ → 结构检查（N ≤ 预算、权重上限）  │     │
 │  │ → 4-seed × X/Z 独立流的 d_search、Q_search = k·d²（09-10 起）  │     │
 │  │ → **先更新 structural_proxy_elites，再做 MAP cell 竞争**      │     │
 │  │ 产出：structural_proxy_elites.json / structural_slow_queue   │     │
 │  │       / promotion_queue.jsonl / all_archive_queue.jsonl      │     │
 │  └──────────────────────────┬──────────────────────────────────┘     │
 │                             │ 6 槽队列：4 结构（high-k / high-d /    │
 │                             │ 2 crowding）+ 2 runtime（named 占用） │
 │  ┌ slow 环（evaluation）─────┴──────────────────────────────────┐     │
 │  │ 每候选：build → SEC 综合（LRC，按候选缓存一次）→ DEM →         │     │
 │  │         X/Z 双基 × 每基 16 shot shards 解码 → Wilson CI → plpr │     │
 │  │ 每个 (候选,p,解码器,基,shard) 是单核子进程作业；           │     │
 │  │ 4 候选 × 2 基 × 16 shards = 128 个 decode work units             │     │
 │  │ build / decode-shard / distance 分别用 3600/1800/1800s watchdog   │     │
 │  │ 超时只重放同一不变请求（原试 + 2 次重试）；全局工作窃取池消费 │     │
 │  │ 固定-shot cell 的任一 X/Z shard 不全 ⇒ measurement_blocked，不入排名 │     │
 │  │ racing：v3.3 冻结为关闭，固定 shots（代码路径保留给变体）      │     │
 │  │ 产出：slow_results.csv / rankings.json / slow_feedback.json  │     │
 │  └──────────────────────────┬──────────────────────────────────┘     │
 │ _audit_turn_execution（仅 --strict-turns）：provider/model 一致、    │
 │   LLM 批次齐整、cycle 状态 → 不达标即抛错；                            │
 │   family 配额达成率只记录不强制（checks.family_quota_shortfall）       │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │ slow rankings
 ┌── reconcile ─────────────┴──────────────────────────────────────────┐
 │ 0. 结构缓存 structural_proxy_elites 跨 turn 进 state（与名人堂独立，│
 │    测得差不会删除结构强码）                                          │
 │ 1. 名人堂更新（按 plpr，d<3 一律不进，cap=12；行内补齐                │
 │    spec/construct_source/rng_seed/source_agent 继承载荷）+ 基线保留   │
 │ 2. 证据行追加 evidence_rows.jsonl（gen 戳；结构坐标随行同写）         │
 │    ——按 gen 去重，保证崩溃续跑幂等                                   │
 │ 3. run_backtest：仅用 propose_gen 之后的行判定假设（前瞻，反双重使用） │
 │ 4. insight_precision：本轮行作留出代，重验已支持规则                  │
 │ 5. 新进名人堂的程序 → 解析假设标签 → bandit 奖励（gen 戳幂等）        │
 │ 6. 新进名人堂的源码 → record_skill 写入 manual/skills                │
 │ 落盘：digest.json（每轮）+ turn_audit.json(complete，仅严格模式)     │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │ turn+1，或 stop
              继续 ←────────┴────────→ final_slow → report → END
```

`campaign.py::run_agentic_cycle` 是 fast/slow 的粘合层；`workflow/graph.py::execute_plan_node` 每轮驱动它一次。跨轮记忆有两条：**结构缓存**（`CampaignState.structural_proxy_elites`，checkpoint 持久化，带完整 spec/construct_source）与 **runtime 名人堂**（`hof_summary`，同样带继承载荷）。每轮的 fast archive 仍从种子 + 这两个池的父代重建，不是持久化对象；slow-aware overlay 只再传 runtime 事实，不再拥有父代顺序。

---

## 4. 上下文装配与工具表

Policy 每轮看到 10 个块，每块是 `<source id kind origin hash>` 元素：

| id | 内容 | 来源 |
|---|---|---|
| S-1 | 手册静态区**索引**（INDEX.md，只列小节，正文靠 `manual_read` 取）+ **完整工具目录（含签名）** | `manual/static/INDEX.md` + `render_tool_docs()` |
| S-2 | 手册动态区（活跃洞见） | `manual/dynamic/insights.md` |
| S-3 | fast 队列 top-12（code_hash/family/n/k/d_proxy/Q） | 上轮 all_archive_queue（缺则回落 promotion_queue） |
| S-4 | slow 排名 top-12（plpr/CI/d_circ_ext/degraded/stopped_by） | 上轮 rankings |
| S-5 | 名人堂 + 基线 | state |
| S-6 | 预算用量与上限 | state |
| S-7 | bandit 建议 | `bandit.suggestion()` |
| S-8 | 上一轮计划及其执行状态 | state |
| S-9 | **结构覆盖度**：按区域的 proposed/measured/censored/best（top-16，只列有提案的区域）+ `lp_nonabelian` 四分支目录计数与 `never_proposed_branches`（零提案只在这一处可见） | `coverage_report()` + `catalog_gaps()` |
| S-10 | **近期反思**（递减窗口：最近 3 轮全文，第 4–10 轮各一行，更早丢弃） | `manual/dynamic/reflections.jsonl` |

S-1 内联全部工具签名是刻意的：模型永远不需要为了知道工具怎么调而先花一次调用去发现。

S-9 回答的是排名块**结构上无法回答**的问题：从未被提案的区域在 S-3/S-4 里不留任何一行，因此"没探过"和"探过但输了"在没有这个块时完全无法区分。censored（racing 提前砍掉）与 measured 分开计数，因为二者要求相反的应对。

但"零提案"只在**枚举得出空间的地方**才是可判定的：区域表由 archive 行反推出来，没提案的区域根本不生成行（`coverage_report` 自己的 note 就这么写），`unmeasured_regions` 说的是"提过但没测过"。真正逐项列出未探分支的只有 `catalog_gaps()`，而它只覆盖 `lp_nonabelian` 的四个语法分支——那是全仓唯一一处把候选空间显式枚举出来的地方。

工具表（`tools/`，装饰器自注册，docstring 即手册单一真源）：

| 工具 | scope | 作用 |
|---|---|---|
| `manual_read` | read | 读手册任意小节 |
| `query_archive` | read | 按 family/限制查 fast archive（含 spec） |
| `query_evidence` | read | 按 code_hash 查全部历史测量 |
| `query_coverage` | read | 结构覆盖度报表 |
| `fingerprint_lookup` | read | 两层结构指纹（WL → pynauty 规范形） |
| `known_code_lookup` | read | 文献已知码等价查询 |
| `acquisition_score` | read | 晋级基线打分 |
| `bandit_status` / `budget_status` | read | 策略臂状态 / 预算 |
| `precheck_draft` | draft | 沙箱试跑草稿程序 |
| `ledger_propose` / `ledger_retract` | policy | 假设提案 / 撤回 |

---

## 5. 学习层

五个组件，全部确定性：

**Insight ledger**（`memory/ledger.py`）——外部化的 campaign 学习。agent **只有提案权**；`supported`/`refuted` 的状态转移**只由回测器**做。被证伪的条目进 graveyard：永久保留、不再进 prompt、谓词哈希封锁再提案。显示 id 为假设期 `H-n`、支持后 `I-n`（同号）。

**前瞻回测器**（`memory/backtester.py`）——ledger 唯一的裁判。反双重使用：判定只用 `gen > propose_gen` 的证据行；要求 scope 内 ≥ 20 行；效应量为谓词真/假两组 `log10(metric)` 均值差；bootstrap 95% CI 必须排除零且符号一致。RNG 由条目 id 播种，重跑逐位相同。

**策略 bandit**（`memory/bandit.py`）——UCB1，5 个臂：`exploit / test_hypothesis / novelty / transfer / wildcat`。奖励 = 该臂在本轮新进名人堂中的贡献份额，从程序源码里的假设标签解析。每臂有 epsilon 下限，不会灭绝。**建议是信号不是命令**——policy 可以带理由偏离。

**两层指纹 + 已知码表**（`memory/fingerprints.py`、`known_codes.py`）——Tier-1 是彩色 Tanner 图（量子比特/X 检查/Z 检查三色）的 WL 哈希；Tier-1 碰撞时升级到 pynauty 规范形，没装则降级并显式标注 backend。新颖性计数因此报为置换等价下的**上界**。已知码表用于**核验侧的重发现判定**，不作为搜索先验注入 prompt。

**技能手册**（`orchestrator/manual.py`）——ACE 式演化 playbook：静态区（工具目录、排名/隔离语义、反馈词汇表）、动态区（活跃洞见 + bandit / digest 快照）、技能区（新进名人堂程序的源码 + 证据）。

**每轮反思**（`orchestrator/reflector.py`）——定性记忆通道。ledger 承载定量学习（必须经前瞻回测才能晋升），但一轮教给系统的东西里有一部分永远成不了"claim"：某个方向不再值得投入、某个构造的行为异于邻居、某个还不能证实的怀疑。反思填这个缺口而**不削弱另一条**：

- **建议性，永不成为证据**——与 evidence 行分开存放，以散文形式给后续轮次，且**会过期**
- **写入时定长截断**（各字段 ≤3 条 ×200 字符，advice ≤400 字符），啰嗦的模型无法撑大后续所有 prompt
- **递减渲染**：最近 3 轮全文 → 第 4–10 轮仅 advice 一行 → 窗口外消失。**无论跑 60 轮还是 600 轮，进 prompt 的量是常数**
- 过期是设计核心：一条观察若在窗口外仍值得记住，它早该经 `ledger_propose` 变成被回测的条目；没变成，就说明它本来不承重
- 失败不影响该轮——测量已提交，建议性通道无权让成功的轮次失败；调用记录进独立的 `reflection_call.jsonl`，严格审计面不动

**Generator 接力便条**（`funsearch.py::_extract_notes`）——一次生成的原始推理有数万 token，永远不可能跨轮传递；能传的是模型**在推理还在眼前时**亲手写下的十行交接（tried / ruled out / next）。这个选择由"当时还记得一切的模型"做，质量高于任何事后摘要，成本为零（同一次调用的输出尾部）。下一轮注入最近 3 轮，约 1k token。

**结构覆盖度**（`memory/coverage.py` + `construction/structure.py`）——把 spec 映射到粗粒度结构坐标：

```text
STRUCTURE_FIELDS = (group_class, group_order, group_order_band, shape_a, shape_b)
structure_key(spec) → "lp_nonabelian/direct_product/12-24"
catalog_gaps() → {dihedral, metacyclic, direct_product, alternating_symmetric} 各自计数
                 + never_proposed_branches
```

这套坐标是**共享词汇**：覆盖度仪表、证据行、ledger 谓词字段三处用同一套，因此"某个结构区域整体更好"这类规律才可能被提出、被回测、被判定。

---

### 5.1 Prompt 设计原则：坏写死、目标极简、路径留白

| 层 | 写法 | 理由 |
|---|---|---|
| **坏（淘汰规则）** | 明确、完备、与校验器对齐 | 违规提案是纯浪费；说清楚不引入方向偏置 |
| **目标（什么算赢）** | 精确但极简，一句话 | 目标若也模糊，模型会用文献先验填空——得到的是先验污染而非探索 |
| **路径（怎么赢）** | 完全留白 | 由测量反馈 / ledger / 反思填充，这才是搜索空间 |

按此原则修正的位置：预算不足写成**代价陈述**而非目标区间；d_proxy 捷径写成 **KNOWN DEAD END**；排序目标一句话（measured plpr 决定，plpr 相近时 larger k 胜）且不解释如何达成；archive 表与 parent 行只展示 `228/288` 这样的**事实**，删掉了原先 "scaling them up (larger ell/m, lift, …) is encouraged" 的配方句。

自有数据支持：d_proxy 被 game（[[42,2,7]] 那批）是"好的标准太具体 + 坏的标准缺一条"的组合结果——修法是补一条坏的；D6 单一栽培的修法是把四个分支的**语法**都展示（约束完备性），而不是告诉模型哪类群更好（那是先验注入）。

## 6. 测量协议

**单候选流水线**（`evaluation/pipeline.py`）：

```text
spec → build_compiled_code (HX/HZ) → 结构检查 + 代理评估
     → 结构不通过即以 structural_reject 返回，不进 SEC
     → SEC 综合（sec/，twobasis_nonint_v1：两基不交错 coloration 电路 + 基内层序认证，按候选缓存一次；lrc_ranked_v1 为历史策略）
     → stim 记忆实验对（X/Z 两基）
     → DEM → 解码器会话 → shot 分片采样
     → Wilson CI → 合并双基 → per_logical_per_round（A5）
     → 证据标注：d_circ_ext、DEM-ISD 距离、degraded_* 标志
```

**噪声 profile**（v3.5，`verification/noise.py` + `sec/stim_builder.py`）：正式协议 = **`uniform_paper_v1`**（OmniQEC 论文 SI C b，单一 p=0.005，**无 idle 噪声**）。LRC 调度 / detector / observable 与 legacy 发射完全一致，只有噪声指令不同；`noise_point=None` 仍表示 legacy 发射（golden 对拍与 `saturate_noise` 结构审计路径逐字节不变）。同一套 lowering 由共享发射器 `sec/stim_builder.py::_emit_circuit` 同时服务 LRC 与 twobasis 两条时间线（追加（七））。快照另冻结 `noise_profile` / `noise_values`，strict 只接受 `uniform_paper_v1` / `[0.005]`。

| profile | 用途 | 噪声放置 |
|---|---|---|
| `uniform_paper_v1`（**正式默认**） | Fig.2 v3.5；`NoiseSweepConfig()` / `uniform()` / `--p` 别名 / CLI `--noise-profile` 默认；`RACING_ANCHOR_P` = 0.005 | 每个 reset（数据制备与每个 ancilla）后 `DEPOLARIZE1(p)`；每层 CX 后 `DEPOLARIZE2(p)`；ancilla 测量前 `X_ERROR(p)`(M) / `Z_ERROR(p)`(MX)；终局数据读出前基共轭翻转（X-memory `Z_ERROR` 后 `MX`，Z-memory `X_ERROR` 后 `M`）；**无 idle**、终段无 `DEPOLARIZE1(data)` |
| `uniform_legacy_v1` | 历史（v3.4 及更早产物）、`tests/golden/` 对拍、`saturate_noise=True` 结构审计；`uniform_legacy()` 显式选用 | 同上的门/测量项 + 逐 tick idle `DEPOLARIZE1(p)`（02 §9.1 第 3 条的选择性省略规则）+ 终段数据额外一次 `DEPOLARIZE1` |
| `si1000_*` / `helios_*` | 平台变体（非正式协议，与 v3.5 不可合池） | 02 §9.2 第 4–5 条 |

**并行模型**（v3.3）：每个 `(候选, p, 解码器, 基, shot shard)` 解码作业和每个距离作业都是独立子进程，由一个全局工作窃取池消费。正式 profile 将 search 与 final 都冻结为每基 `shot_shards=16`；标准 4 候选 slow 队列因此产生 `4×2×16=128` 个解码 work units（resident baseline 及 distance 作业另计）。NSCC 整节点申请 128 CPU，其中工作池用 `slow_workers=127`。这能让主解码波次接近满核，但 LLM、build、distance 与 straggler 尾部本来就不会填满 128 核，所以 PBS 的全作业累计 CPU 平均不应被解读为纯 decode 利用率。此池仍是**单节点**：用 `subprocess.Popen`，无 MPI；多节点 PBS 分配会让额外节点闲置。**v3.4 更新**：search 仍是每基 16 shards，但终局改为每基 **`final_shot_shards=128`**（200k shots → 1562 shots/shard），终局 work units = 入围数 × 2 × 128，远多于 127 个 worker。

**分类 watchdog 与重试**（v3.3）：build/DEM 作业上限 `build_timeout_s=3600`，每个 decode shard 上限 `decode_shard_timeout_s=1800`，距离作业上限 `distance_timeout_s=1800`，`timeout_retries=2`。重试只针对 timeout，且每次重放完全相同的预注册 request（同 candidate/basis/shard/seed），最多共 3 次；确定性子进程错误不借重试隐藏。旧 `circuit_timeout_s` 只是读旧 artifact 的兼容字段；v3.3 snapshot 显式冻结三个分类值与重试次数。**v3.4 更新**：decode shard 的有效 watchdog = `max(decode_shard_timeout_s=1800, decode_shot_budget_s=8 × 该 shard 的 shots)`（search 10,000 s / final 12,500 s），`timeout_retries` 冻结为 **0**（超时不重放），新增 `transient_retries=1`（只对 worker 被杀 `subprocess_error` / DEM 丢失 `dem_load_error` 同请求重放一次）；见 §10「2026-09-02 追加（五）」。

**固定-shot 完整性门**（v3.3）：只有配置中所有 basis 都存在、每基恰好完成 16 个 status=`ok` 的 shard，且合并 shots 恰好等于预注册 shots，该 decoder cell 才能发布 PLPR。任一 X/Z shard 丢失、超时或不完整时，部分计数只留在 `partial_combined` 供基础设施诊断，正式 `combined` 清空，候选与 slow queue 进入 `measurement_blocked`，不排名、不用更容易的新码补位。距离认证是辅助证据：distance timeout 记为 `uncertified_timeout`，但不抹掉已完整的 X/Z PLPR。**v3.4 更新**：若所有必需基都返回、且唯一的缺陷是 watchdog 超时，候选与结果记 **`decode_infeasible`**（终态：不排名、不入终局、不重排队、队列仍 `ok`、不停止该点）；任何其它不完整仍是 `measurement_blocked`。终局每基须恰好 128 个 `ok` shard。

**Racing**（`evaluation/sampling.py`）：**v3.3 冻结为 `enabled=false`，固定 shots**，strict validator 强制该状态。算法与代码路径完整保留给后续 protocol variant：shard 流预注册为 `(seed, shards_max)`，racing 只决定消费深度，从不改变 RNG 协议；`racing_decision` 是累计计数与静态阈值快照的纯函数，淘汰与 shard 完成顺序无关。

**解码器**（v3.3，配置沿用 v3.2）：全流程**单一** BPOSD，配置 `product_sum / max_iter=50 / osd_cs / osd_order=10`（`preset="2d"` 是兼容标签，ldpc 2.4.1 不接受该参数，adapter 只传上述四个有效字段）。搜索期、runtime 名人堂、固定 T 终局审计用同一份配置；禁止 LSD/Beam fallback、禁止多解码器逐码取最小 plpr、禁止逐 family 调参。仓库里只剩这一份 BPOSD 配置（`config.BposdDecoderConfig` 默认值，`qualification.BPOSD_REFERENCE` 与 CLI `--osd-order` 默认都指向它）。strict validator 逐层核对 snapshot / 每 turn slow run config / 每条 search ranking 行 / final run config / 每条 final 行的 `decoder == "bposd"` 与四个字段，并检查 `ldpc==2.4.1` 与 BPOSD backend 已解析；调用方传 `--decoders lsd` **不能**把别的协议授权成本协议。

**final_slow**（v3.3，入围与轮数规则沿用 v3.2 §4.6）：入围 = **3 结构**（high-k 端点 / high-`d_search` 端点 / crowding 代表）+ **1 runtime**（BPOSD 名人堂第一名）【2026-09-10 起 **4 结构 + 2 runtime**（`final_structural=4 / final_runtime=2`），与每轮慢队列同配比】，hash 去重、重叠或池不足时按各自冻结顺序回填；resident baseline 另测，不占名额。全部用 fresh shots、同一 BPOSD、每基 16 shards（**v3.4：每基 128 shards**，`final_slow_node` 读快照 `final_shot_shards`）、**固定 `T_final=12`**——旧实现跑的是 `T=max(3, d_proxy)` 却标成 `T=d`，而 `d_proxy` 是启发式搜索值不是认证距离，既误标了测量又让每个候选的电路深度由搜索自己挑。轮数的唯一真源是 protocol snapshot 的 `rounds_policy.{final, final_rounds}`；`verification.rounds` 若同时存在必须严格相等，缺字段 fail closed。final 用独立 RNG namespace `final_confirmation_v1`（base seed ≥ 1e12），与搜索期的采样键不可能重叠——"fresh shots" 因此是可审计事实而非口号。

**结构保留通道**（v3.3 沿用 v3.2 §1/§4，`ranking.reduce_structural_proxy_elites`）：每个预算点独立维护最多 16 条 `(k, d_search)` proxy-Pareto elite。支配关系只看 `k` 与 `d_search` 两个绝对量；`n_total`、budget utilization、family、check weight、degree 全部记录但**不作主轴**（把 `n↓` 放上主轴等于重造"小码更容易留下"的选择压力）。非支配分层后每层必保 max-k 与 max-`d_search` 端点，其余按二维 crowding 距离填充；只有 `(k, d_search)` 完全相同才依次按更小 `n_total`、更高 `Q_search`、`code_hash` 打破平局。该 reducer 不读 plpr / d_eff / d_cap / `degraded_*` / 电路距离中的任何一个。

**三重资格**（v3.3 沿用 v3.2 §4.1）：`proxy_valid`（build/shape/k/proxy 全部完成，完整 n/k/d_search/Q 可用）、`runtime_screen_passed`（= proxy_valid ∧ d_search≥3 ∧ `d/√n` trust screen）、`structural_eligible`（= proxy_valid ∧ 结构合法 ∧ k>0 ∧ d_search≥3 ∧ 继承载荷可重建）。`FastResult.valid` 只是 `runtime_screen_passed` 的兼容别名。trust screen 在 `n_data=288` 上会拒掉 `d_search=4`（0.236 < 0.25）——那正是本通道要保住的 high-k 端点，所以它只影响旧 MAP/runtime-acquisition 通道，且失败候选保留完整 proxy 记录（旧实现把它写成一条 error，记录直接消失）。

---

## 7. 产物树

```text
runs/<campaign>/
├── campaign_spec.json          # 冻结的 Fig.1 intake
├── protocol_snapshot.json      # 冻结协议（见 §8）
├── resolved_versions.json      # 依赖版本快照
├── checkpoint.sqlite3          # LangGraph SqliteSaver（durability="sync"）
├── evidence_rows.jsonl         # 全 campaign 证据行（gen 戳 + 结构坐标）
├── ledger.json                 # insight ledger（含 graveyard）
├── bandit.json                 # UCB1 状态
├── manual/                     # static / dynamic / skills 三区
├── turns/turn_NNNN/
│   ├── plan.json               # 执行用计划（policy/proposal_count/tool_calls_used/...）
│   ├── digest.json             # 本轮摘要 + learning 块
│   ├── turn_audit.json         # 严格审计（仅 --strict-turns 才写；execution_complete → complete）
│   ├── token_usage.json        # 本轮成本：by_role（orchestrator/reflector/generator）+ total
│   ├── reflection.json         # 本轮反思（其调用记录另存 reflection_call.jsonl）
│   ├── orchestrator/           # llm_calls.jsonl / llm_response_NN.txt / turn_plan.json /
│   │                           # tool_transcript.json / context_blocks.json / events.log
│   ├── parent_selection.json   # 本轮 12 条父代的来源与渲染顺序 + 继承校验错误
│   └── cycle/
│       ├── agentic_cycle_summary.json
│       ├── slow_queue.jsonl            # 实际测的 4 槽队列
│       ├── slow_queue_composition.json # 槽位来源、named 是否占用、structural_slot_shortfall
│       ├── fast/               # 以下仅列关键文件。archive.json / promotion_queue.jsonl /
│       │                       # all_archive_queue.jsonl / structural_proxy_elites.json /
│       │                       # structural_slow_queue.jsonl / funsearch_summary.json（严格
│       │                       # 审计必需）/ generation_notes.json（接力便条，只落盘不回灌，
│       │                       # progress.csv 的 gen_notes 来源）/ generations/g0001/llm_raw.jsonl
│       └── slow/               # slow_summary.json / slow_results.csv 与 .jsonl /
│                               # rankings.json / slow_feedback.json /
│                               # distance_evidence.json / candidates/
├── final_slow/
│   ├── T12/                    # 终局固定 T=12 测量（全体候选同深度）
│   ├── queue_T12.jsonl         # 输入队列（3S+1R + baseline）
│   ├── final_shortlist.json    # 入围名单与 selection_channels
│   └── final_slow_summary.json # 含 final_measurement_audit 与 pending_extensions（= §11 R3）
├── campaign_report.json/.md
├── structural_proxy_elites.json/.csv  # 结构账本（d_search_proxy 列名明写是代理值）
├── runtime_hof.json/.csv              # runtime 账本
├── fixed_T_final_audit.json/.csv      # 固定 T 终局，含 selection_channels
├── summary.csv                 # 每个码的最优测量（按 plpr 排序）
└── progress.csv                # 每轮一行的进度视图
```

**三份分栏报告**（v3.3 沿用 v3.2 §4.6）：结构账本、runtime 账本、固定 T 终局各自成文，同一个码可同时属于 structural 与 runtime（`selection_channels` 保留双标签），两栏互不覆盖。`d_search` 在所有列名里都带 `proxy` 字样：这些文件会被当码表读，把启发式距离印成 `d` 就是一个 `[[n,k,d]]` 声明。

**两个 CSV 的分工**：`summary.csv` 回答"哪些码好"，`progress.csv` 回答"跑得怎么样"。后者的关键列：`policy`（LLM 在开车还是兜底接管）、`measured/censored`、`best_plpr_cum` 与 `vs_bb_cum`（已剔除每轮重测的基线，`<1` 才算赢过 BB）、`mix_*/strat_*/explore_temp`（策略漂移，自主性的直接证据）、`orch_llm_calls`（突然升到 5–6 次 = 撞上格式/校验修复预算）、`tok_total` 与 `cost_usd`、`reflection`、`gen_notes`。逐角色的 in/out/reasoning/cache_hit 细分**不进 CSV**，留在同一轮的 `token_usage.json` 里——每轮只看两个数，细分要看时再打开那份文件。

**成本口径**：token 与金额**同时**统计，互不替代。

| 字段 | 在哪 | 含义 |
|---|---|---|
| `tok_total` | progress.csv | 即 `token_usage.json` 里 `total.total_tokens` = `input + output`，计费总量 |
| `cost_usd` | progress.csv 与 token_usage.json 同名 | provider 报告的**实际计费金额**（目前只有 OpenRouter 提供） |
| `input` / `output` | token_usage.json 的 `total` 与 `by_role.<role>` | 输入 / 输出 token；CSV 里没有这两列 |
| `reasoning` | 同上 | 推理 token——是 `output` 的**子集**（DeepSeek/OpenAI 计在 completion_tokens 内，Anthropic 的 thinking 按 output 计费），**不可与 `output` 相加** |
| `cache_hit` | 同上 | 命中缓存的输入 token |
| `reasoning_complete` | token_usage.json 的 `total` | 是否每一次调用都上报了 reasoning 计数。订阅席位服务的 Claude 调用根本不返回 `completion_tokens_details`，把它记成 0 会让"没推理"和"没上报"变成同一个数——所以未上报就是未知，并在这里标出来 |

两套口径并存的理由：OpenRouter 每次调用返回 `cost`，那是**当时按当时价格实际扣的钱**，比"token × 事后价目表"准；而直连 provider 不报金额，那里 `cost_usd` 恒为 0，仍然靠 token 数乘单价。同一份成本记录因此能横跨两类 provider 做对比。

每轮聚合从三份已落盘的调用记录**重算**而非累加，因此续跑重入不会重复计数。

---

## 8. 冻结协议与断点续跑

`build_protocol_snapshot()` 冻结 policy **不可触碰**的一切（`workflow/state.py`）：`protocol_profile`（v3.3 = `bposd_structural_v3_3`，协议身份本身）、`sec_policy_id`（v3.5 SEC 起 = `twobasis_nonint_v1`，并新增顶层 `sec_candidates`=16 / `sec_cert_iters`=400，`verification.sec_policy` 四字段含 `sec_cert_seed`=0；strict 只接受这组值，非 strict 选 `lrc_ranked_v1` 给 `protocol_profile` 追加 `+seclrc_ranked_v1`、同 id 改 K/iters/seed 追加 `+secK{K}I{iters}S{seed}`）、**prompt 内容哈希**、噪声 profile 与 anchor_p（v3.5 起另有顶层 **`noise_profile`** 与 **`noise_values`** 两键；strict 要求 `uniform_paper_v1` / `[0.005]`，非 strict 的非 paper profile 给 `protocol_profile` 追加 `+noise<profile>`）、`final_p_values`、`rounds_policy`（search/final 各自的 fixed_Tc 与轮数、`final_rng_namespace`）、`structural_retention`（reducer id、proxy seeds/effort、elites cap、slow 槽与终局 **4S+2R**（2026-09-10 起；原 3S+1R）、8S:4R 父代）、完整 `verification` 与 `racing` 配置、`final_shots`、**`shot_shards`**、`max_turns`、`tool_budget`、`proposal_count`、`generations_per_turn`、`promote_top`、`slow_workers`、`feedback_mode`、`strict_turns`、`required_provider/model`、`rng_seeds_per_program`，外加三个运行期不可变的键：`sec_synthesis`、`sampling_source`（恒为 `noisy_circuit`）、`fixed_program_count`（恒等于 `strict_turns`）。`verification` 内的 `build_timeout_s/decode_shard_timeout_s/distance_timeout_s/timeout_retries`（v3.4 再加 `decode_shot_budget_s/transient_retries`）同样是协议字段；v3.4 快照另有 **`final_shot_shards`**（`shot_shards` 只指 search）。

`promote_top` 必须等于 `structural_retention.slow_candidate_cap`，否则 `build_protocol_snapshot` 直接抛错——队列容量与它自称遵循的 S:R 配比不允许各说各话。`plan.promote_top`、`len(plan.promotions)` 与 CLI override **都不能**改这个 cap：出现漂移时 `execute_plan_node` 抛错，而不是截断后继续。

v3.3 的快照与 v3.2 必然不同（profile、`shot_shards`、分类 watchdog 与重试策略），而 shard 数还直接改变 index-derived RNG 流；因此 **v3.2 checkpoint 一律不能用 v3.3 `--resume`**。当时的 Claude launcher 用带 `_v33` 后缀的新 OUT_DIR，既保留旧产物，也把升级变成一次干净新跑。同一 v3.3 run 也不允许中途改 shard 数或 watchdog 后续跑。v3.4 同理：快照新增 `final_shot_shards` 与 `verification.{decode_shot_budget_s,transient_retries}`，`timeout_retries` 默认由 2 改 0，因此 **v3.3 checkpoint 一律不能用 v3.4 `--resume`**；v3.4 launcher 改用 `_v34` 目录。v3.5 同理：快照新增 `noise_profile` / `noise_values`，v3.4 快照缺这两键即按协议不匹配拒绝（不会被静默接受），因此 **v3.4 checkpoint 一律不能用 v3.5 `--resume`**；v3.5 launcher 改用 `_v35` 目录。v3.5 SEC（追加（七））同理：快照 `sec_policy_id` 由 `lrc_ranked_v1` 变为 `twobasis_nonint_v1` 并新增 `sec_candidates` / `sec_cert_iters` 与 `verification.sec_policy`，SEC 落地前首跑的任何快照（含同日早些时候 SEC 仍为 LRC 的 v3.5 噪声版）按协议不匹配拒绝，**不能 `--resume`**，须换新目录。

其中 `--generations-per-turn` 与 `--feedback-mode` 是 `run-campaign` 的普通旗标：续跑时改动任意一个都会让快照对不上，`--resume` 直接拒绝。

`--resume` 时 `validate_resume_protocol` 比对快照，**任何差异一律拒绝续跑**——差异键会打印出来。快照记录的是**将被实际执行**的协议：例如 worker 数解析为 1 时 racing 无法工作，快照里 racing 就记成 disabled，而不是记请求的标志位。

`strict_turns` 打开后：禁止 fallback 轮、要求 provider/model 逐调用一致、要求 LLM 批次齐整、要求恰好 `turn_0000..turn_00NN` 全部 audit 通过才允许进 final_slow、final 测量的 p/shots/T 逐项复核。

**注意一个真实的口子**：`prompt_ids()` 只哈希 `orchestrator/prompts/*.md`。Generator 的 system/user prompt 是在 `funsearch.py` 里**用代码拼的**，不在快照内——改 generator prompt 不会使协议快照失效。这既是便利（改 `plan.py` 的报错文案不打断 22 轮进度）也是风险（改 generator 措辞会静默改变可比性）。列入 §11。**2026-08-26 就真发生了一次**：`catalog_summary()` 增补了一句二面体索引约定（文献的 S3 = 本目录 `D3`、其 D10 = 本目录 `D5`），经 `construction/registry.py:213` 的 `format_contract_lines` 原样进入 generator system prompt（`funsearch.py:581`），而协议快照对此**零反应**——跨该日比较 run 时须人工标注，见 §10 末节与 §11 R1。

---

## 9. 运行方式

**本地/CI**：`--provider dry_run` 走 FallbackPolicy，全管道可跑通、无需 key。

**一个预算点的正式协议**（`bposd_structural_v3_5`：噪声 `uniform_paper_v1` p=0.005（论文 SI C，无 idle）、BPOSD 单解码器、racing 冻结关闭、固定 shots、search 每基 16 shards / final 每基 128 shards、decode-shard watchdog = max(1800 s, 8 s/shot × shots)、超时不重放、瞬态错误重放一次、T_c=12）：

```bash
python -m omniqec_agentic.cli run-campaign --budget 288 --out <dir>_v35 \
  --noise-profile uniform_paper_v1 --p 0.005 --turns 60 --proposal-count 24 --rng-seeds-per-program 20 \
  --provider deepseek --decoders bposd --final-shots 200000 --shot-shards 16 \
  --final-shot-shards 128 --decode-shot-budget-s 8 \
  --build-timeout-s 3600 --decode-shard-timeout-s 1800 \
  --distance-timeout-s 1800 --timeout-retries 0 --transient-retries 1 \
  --strict-turns --slow-workers 127
```

旧协议的 `--racing ... --decoders lsd` 组合在 v3.5 中仍为退役：`--strict-turns` 下两者都会在
**启动时**被拒绝（`cli.py` 与 `build_protocol_snapshot` 双重拦截），不会烧完 60 轮再被
外部 validator 打回。

**集群**：v3.5 当前的生产入口只有 NSCC Claude 订阅臂。它请求单节点 128 CPU / 100 GB，使用 127 个 slow worker，并以 24h PBS `afterany` 链跨越 walltime 墙。100 GB 是执行资源申请而非科学协议字段；正式全网格前仍须用**不带 `FINAL_SHOTS_OVERRIDE`** 的 b360 canary（终局阶段约 3.5 h，127 个并发 decode 子进程各持最大入围码的 DEM + BPOSD）的作业级峰值内存确认余量。其他平台/臂的 launcher 仍绑定 v3.2 或更旧协议；在同样接入 search 16 / final 128 shards、按 shot 预算的 watchdog 与重试参数、`decode_infeasible` 语义、显式 `--noise-profile uniform_paper_v1 --p 0.005` 与 v3.5 validator 之前，它们的数字**不能与 v3.5 合池**；v3.4 产物（idle 噪声）同样不能（R21）。

| 路线 | 脚本 | 执行模型 |
|---|---|---|
| NSCC/PBS + Claude v3.5 | `hpc/nscc_claude_grid_v35.pbs`（单点或链式续跑，job `omni_claude_v35`，每次 run-campaign 显式 `--noise-profile uniform_paper_v1 --p 0.005`，启动自检 `V35_DEFAULTS_OK`，含 `DEFAULT_SEC_POLICY.sec_policy_id == 'twobasis_nonint_v1' == FORMAL_SEC_POLICY` 与 K=16 / iters=400）、`hpc/submit_claude_v35_grid.sh`（每个预算点独立链）、`hpc/claude_v35_status.sh`（显示快照 `noise_profile`）、`hpc/claude_v33_diagnose.py`（版本无关）、`hpc/nscc_cliproxy_shared.sh` | 唯一已接入 v3.5（v3.4 几何：search 16 / final 128 shards、8 s/shot 预算 watchdog、retries 0/1；噪声 `uniform_paper_v1`）与 validator 的生产路线；CSV 在 validator 之前导出；`measurement_blocked` 立即停止该点，不重进 Generator；`decode_infeasible` 是终态、**不停止**。v3.4 三件（`*_v34.*`，头部标 HISTORICAL）与 v3.3 三件（`*_v33.*`）保留为历史 |
| NSCC/PBS 通用入口 | `hpc/fig2_campaign.pbs` | 尚未升级为 v3.3/v3.4/v3.5 生产入口；不能用它声称 v3.3/v3.4/v3.5 结果 |
| 电光云/Slurm | `hpc/dgcloud_fig2_one.sbatch`、`hpc/dgcloud_fig2_grid.sbatch`等 | 未接入 v3.3/v3.4/v3.5，仍与旧代码/协议快照绑定 |
| 无调度器 Pod 平台 | 平台专用的单点、看护、臂间串行脚本 | 未接入 v3.3/v3.4/v3.5，仍与旧代码/协议快照绑定 |

状态视图：v3.5 Claude 臂用 `hpc/claude_v35_status.sh`，它同时显示 PBS 累计 CPU 平均、审计轮数、snapshot 的 search/final shard 数与 `noise_profile`（非 `uniform_paper_v1` 打 `⚠noise=`）、逐 run 目录汇总的 `decode_infeasible` 候选数与近期错误；`hpc/claude_v34_status.sh` 只看 `_v34` 目录，`hpc/claude_v33_status.sh` 只看 `_v33` 目录。旧臂仍用 `hpc/nscc_status.sh`、Pod 平台的状态脚本与 `hpc/status_all.sh`；这些旧视图不代替 v3.5 validator。

**Claude 订阅席位（CLIProxyAPI）**：这条臂用 `--provider openai` 打 CLIProxyAPI 的 OpenAI 兼容端点（`llm/openai_compat.py`），**不是** `--provider claude`——anthropic SDK 不在 `hpc/requirements_fig2_nscc.txt` 里，那条路径从未被执行过。并行跑必须走 `nscc_cliproxy_shared.sh`：一个订阅 grant 只能有一个 refresher，多个实例各自刷新会被上游的 refresh-token 复用检测判成盗用而吊销整个 grant。v3.5 的 `submit_claude_v35_grid.sh` 每点只建一条独立接力链，并在提交前拒绝同 `RUN_TAG`/预算点的重复作业；`nscc_claude_grid_v35.pbs` 再用 `$OUT_DIR/.nscc.lock` 做运行期守卫。当时的上机操作手册（v3.3–v3.5）是账号本地文件，不在开源分支中。

**LLM 网关**：NSCC 防火墙 TLS-reset 所有 AI API 域名，但 `*.workers.dev` 可达。`hpc/worker_deepseek/` 是一个 Cloudflare Worker，把 `/v1/chat/completions` 与 `/v1/models` 转发到 api.deepseek.com。Worker **不持有任何凭据**；密钥只在 `~/.config/omniqec/deepseek.env`（chmod 600）。可选的 `GATEWAY_TOKEN` 作为 URL 路径段，客户端代码零改动。部署手册是站点专用文件，不在开源分支中。

两个生产环境才暴露的坑（本地 `wrangler dev` 复现不出）：请求体必须 `arrayBuffer()` 缓冲后再转发（流式会让运行时发 chunked，DeepSeek 背后的 CloudFront+ELB 会卡死，每个 POST 恰好 30s 挂掉）；响应必须删掉 `content-encoding`（运行时已解压，但头还声称 br）。

---

## 10. 本轮更新（2026-08-17 → 08-22）

起因是一个具体问题：**Mitten 码在 `lp_nonabelian` 语法覆盖范围内，却从未被搜出来。** 把已知码作为先验注入 prompt 会破坏自主发现的立论，因此改动全部落在**机制**上——让 agent 有能力自己找到那片区域，而不是被告知答案。

### 自主性机制（6 项）

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| 结构坐标 | `construction/structure.py`（新） | 搜索空间没有名字，"某类群更好"这种规律无法表达 |
| 覆盖度仪表 + S-9 块 + `query_coverage` | `memory/coverage.py`（新）、`orchestrator/loop.py`、`tools/queries.py` | 零提案区域在任何排名里都不留行，与"探过但输了"无法区分 |
| ledger 谓词字段扩展 | `memory/ledger.py` | 结构级假设提得出来也回测不了 |
| family_mix → 整数配额 | `workflow/graph.py`、`funsearch.py` | family_mix 只作为一行散文"注意力"进 prompt，没有任何东西分配批次，policy 实际上无法引导 |
| parent 槽 round-robin 交错 | `memory/archive.py` | 构造了 12 个 parent 只渲染前 6 个，多样性槽被顺序截断全部砍掉 |
| explore_temperature | `orchestrator/plan.py`、`memory/archive.py` | 利用/探索的强度是写死的常数，不是 agent 的决策 |

配套：`SEED_LP_METACYCLIC` 种子（群 `C7:C3(2)`，实测 [[42,4,5]] 权重 5，**非文献码**）；`registry.py` 的 `lp_nonabelian` 语法示例从只有 D4 扩到四个分支（二面体/亚循环/直积/交错对称，注意元组元数不同）；racing 提前停止的行在 prompt 里显式标注 `racing_stopped_early(low-precision, not a settled result)`。

审计结论（14 个 agent，41 项发现，0 项被驳回）确认了基准盲化是真的——D6 单一栽培是**涌现**的，不是泄漏。IBM 2026 测得同类现象下 57% 的模式坍缩；Liu & Marquardt 2026 把 WL 指纹用作**适应度新颖性项**，而 OmniQEC 此前只把同一个量用作去重闸门。

### 可靠性

- **LLM 超时**（`llm/retry.py` 新增）：`call_with_deadline` 守护线程 + `connect_timeout_s` / `total_deadline_s`；四个客户端全部 `max_retries=0`（SDK 自带重试把超时预算翻了三倍）。
- **`--circuit-timeout-s`**：大预算下 agent 会合理地提出高距离码，600s 默认值会让整轮候选全超时，而 `slow_queue_status` 只有在**全部**超时时才返回 "timeout" —— 严格审计随即拒绝该轮，确定性重放烧掉整条作业链。按预算分档为 900/1800/3600。
- **计划校验报错可执行化**（`orchestrator/plan.py`）：偏离 acquisition 基线的晋级必须给理由，但报错只说规则不说 JSON 形状，模型连续 4 次修复都没猜出要写 `rationale: {text, cites}` 字段，直接杀死一个跑到第 22 轮的 campaign。现在报错自带可照抄的例子与第二条出路（"改为晋级基线头部，无需理由"）。**规则一点没放松。**

### 可观测性

- `progress_csv.py`（新）+ 两个测试：每轮一行，跑到一半也能导；已接入两个集群的收尾。
- `nscc_status.sh`：新增 `csv` 子命令与 conda 自激活；修掉 `archive` 子命令落进 `BUDGET` 变量导致日志 glob 落空、**可能把正在写的日志一起归档**的 bug。

### 效果

| 阶段 | b288 最好成绩 |
|---|---|
| 首次（beam+lsd，600s 超时） | 3 轮 |
| 本轮（lsd，3600s 超时，racing max 30000） | **22 轮** |

---

### 2026-08-22 追加：预算压力、反思记忆、成本统计

起因是 b288 结果的两个现象：**赢的码普遍只用掉一半预算**（[[78,4,8]] 花 156/288），且 **k 和 d 分裂**（赢基线的全是 k=4 的 LP 码，k≥16 的 hgp 全是 d=3–4）。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| 预算代价陈述 + KNOWN DEAD END + 排序一句话 | `funsearch.py` | prompt 里预算只是上界，没有一个字说明未花掉的预算买不到东西；也从未提过 k 的偏好 |
| archive 表 / parent 行显示 `228/288` | `funsearch.py`、`memory/archive.py` | Q=k·d²/n 尺度无关，表里小码与满预算码看起来同样体面——浪费不可见就不会被修正 |
| 删除 parent 块的 scale-up 配方句 | `funsearch.py` | 它直接点名该施加的变异（larger ell/m, lift, group），属于路径层的先验注入 |
| Reflector 每轮反思 + 递减窗口 S-10 | `orchestrator/reflector.py`（新）、`prompts/reflector.md`（新）、`workflow/graph.py`、`orchestrator/loop.py` | 系统只有定量通道（ledger→回测），一轮教的定性内容（哪条路不再值得走）无处存放，每轮从零开始 |
| Generator 接力便条 | `funsearch.py`、`workflow/graph.py` | generator 每轮消耗上万 reasoning token，然后**什么都不留下** |
| 逐调用刷新 usage 快照 | `orchestrator/loop.py` | `client.info()` 只在循环前取一次，而 usage 是"上一次调用"的量——b144 的 llm_calls.jsonl 里 `model` 块**根本没有 usage 键**，planner 的成本完全不可见 |
| 每轮 token 汇总 + budget 累加 + 4 个 CSV 列 | `token_usage.py`（新）、`workflow/graph.py`、`progress_csv.py` | `budget.tokens_used` 硬编码为 0，从未累加 |

明确**不做**的：Q = k·d²/n 不动（用户定案：该指标是必须的）——**2026-09-10 用户改定案：Q 改为 k·d²，去掉 /n**（`ranking.QUALITY_METRIC_ID = kd2_v1`，进协议快照 `quality_metric`，validator 核对；旧快照无此键 → 不能续跑/不能验证）；plpr 纵轴不动；`proxy_n_min` 硬下限不做（一刀切会杀死用于假设检验的小码）；reasoning 全文不存（只记 token 计数）。**2026-09-10 下午再定**：慢环晋级门槛 `ranking.MIN_PROMOTION_DISTANCE` 3 → 5（v36 老数据：d_search ≤ 4 占 34–60% 慢测、从未进最好码 1.5 倍内）；慢队列 4S+2R = 6/轮、终局 4S+2R；fast loop 每轮 24 提案 × 4 代 × 40 种子（GPT 慢测码 0% 来自 rng 种子 > 0、Claude b98 以上 6–7%、DeepSeek 85%，所以加代数才能扩确定性臂的广度，加种子只对随机化臂有效）。运行手册 `nscc_gpt_v36_serial_runbook.md` §14.2。**2026-09-11 再定**：慢环早停 `pilot10x_v1`（pilot 2 000 枪/基、250 枪一片、置信下界 ≥ 名人堂最好码上界 × 10 即停、无基线臂；终局不早停），`config.formal_racing()` 冻结、validator 逐字段核对、`--no-early-stop` 为变体；手册 §17。

三家 provider 的 reasoning 可得性（决定了上一条）：DeepSeek 给完整原始 CoT，Claude 给摘要化 thinking 块，ChatGPT 的原始 CoT **不通过 API 提供**（Responses API 只能拿服务端摘要）。token 计数三家都有，因此成本统计不受影响。

### 2026-08-23 追加：OpenRouter 统一路由与两项降本

**GPT / Claude 臂统一经 OpenRouter**，不新增 provider：`--provider openrouter` + `OPENROUTER_MODEL=openai/gpt-5.6-sol | anthropic/claude-opus-5`。记录下来的 model id 本身标识哪条臂，也正是严格审计所钉的值。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| `resolve_model_name(provider, config)` | `llm/factory.py`、`cli.py` | `LLMConfig()` 带的是 DeepSeek 的 env 名，`config.model` 对任何 provider 都返回 `deepseek-chat`；它被当作 `required_model` 后与逐调用审计**结构上必然不等**——非 DeepSeek 臂在第 0 轮全灭 |
| 输出上限 16000 → 64000 | `llm/openrouter.py` | 常量名带 DeepSeek 却对所有模型生效；b144 一次 generator 调用就用了 19,336 completion token，会被截断 |
| `cost_usd` 贯通统计链 | `llm/usage.py`、`token_usage.py`、`progress_csv.py` | 只有 token 没有金额；OpenRouter 每次调用返回实际计费额，比事后乘价目表准 |
| 记录 `served_model` / `served_upstream` | `llm/openrouter.py` | 路由器可能换模型或换上游，而审计只校验请求方，替换不留痕迹 |
| **A：Anthropic 块级缓存** | `llm/openrouter.py`、`orchestrator/loop.py` | Anthropic **不自动缓存**，必须显式标 `cache_control`；orchestrator 每轮重发整段对话（opus-5 实测 10 次往返、127k 输入），而所有 Claude 调用的 `cache_hit` 全是 0 |
| **B：批量工具调用** | `orchestrator/loop.py`、`prompts/orchestrator_system.md` | 互不依赖的只读查询被拆成 10 次往返，每次重发全文并重付一段推理前奏（输出的 34%） |

**A 的关键细节**：Anthropic 在**内容块边界**匹配缓存前缀。把整段对话塞进一个 block 并在末尾打标记几乎缓存不到东西——block N+1 与 block N 字节不同，只有独立的 system block 会命中（opus-5 实测：98k 输入仅命中 7k）。正确做法是**每个对话片段一个 block、标记只打最后一块**，于是前 N 块构成下一次调用的逐块前缀。实测第 2 次调用 14,467 输入命中 14,323（成本 −90%）。客户端签名探测（`_accepts_segments`）保证不支持该参数的客户端走原路径。

**B 的三条护栏**（均有测试）：单工具信封逐字不变；批量**不额外增加** tool_budget；**不强制批量**——参数依赖前一结果的查询仍可逐次发。系统提示只陈述机制与约束，不下达"应当批量"的指令（沿 §5.1 的路径留白原则）。

**降本实测**（b288，24 提案，2 轮）：gpt-5.6-sol 往返 5→3 而调用的工具 4→12 个、成本 −29%；claude-opus-5 turn 0 的 orchestrator 命中 0→20,363（占输入 77%）、该部分成本 −63%。能力维度（policy 未退回 fallback、measured 码数、覆盖 family 数、best_plpr）无退化。

**部署约束**：B 改了系统提示 → `prompt_ids` 变 → 协议快照变 → **旧 checkpoint 无法续跑**；A 只在客户端传输层，不进快照。

### 2026-08-24 追加：集群脚本与验收协议对齐

起因是三处「跑的协议」与「验收/观测的协议」脱节，都只在集群上才暴露。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| racing 四值 + decoder 集合改为验证器参数，由作业脚本原样传入 | `hpc/validate_fig2_campaign.py`、`hpc/fig2_campaign.pbs` | 验证器把 racing 硬编码成 20000/50000，而 PBS 支持 `-v RACING_MAX=...`——§9 记录的 b288 最佳跑法（`--racing-max 30000 --decoders lsd`）跑满 60 轮后必被拒；同时 decoder 集合两个方向都不校验，而它才是真正让数字不可比的那个变量 |
| 两个 CSV 移到验证之前导出 | `hpc/fig2_campaign.pbs` | `set -e` 让被拒的 run 跳过导出，而顶部的「已完成则秒退」守卫又让重交直接 exit 0——被拒的 run 永远拿不到 `progress.csv` |
| 非 DeepSeek 臂不再猜默认模型名 | `hpc/fig2_campaign.pbs` | fallback 写着 `gpt-5.5` / `claude-opus-4-8`，与实际在跑的 `gpt-5.6-sol` / `claude-opus-5` 漂移；它会被钉进 `required_model`，正是 README_LLM_ROUTES §5 第一行那个「第 0 轮全灭」的失效模式 |
| 状态脚本的连通性 grep 跟上 probe_llm | `hpc/nscc_status.sh` | 找的是 `deepseek connectivity`——只有电光云的内联探针打这行，NSCC 侧换成 `probe_llm` 后该行永远空白，归档判据也漏掉一整类失败日志 |

验证器现在对覆盖过的 run 打印 `[PROTOCOL VARIANT: ...]`，与 `protocol_snapshot.json` 互为佐证。提交时可覆盖的三个变量见 `hpc/README_FIG2_NSCC.md` §6.1。

### 2026-08-25/26 追加：第三个平台与 Claude 订阅臂

代码本身没动，动的全在 `hpc/`：多了一个没有调度器的平台，和一条不经网关的 provider 路线。两者都只在集群上才成立，所以记在这里而不是设计稿里。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| 无调度器 Pod 平台：单点脚本 + 看护 + 臂间串行 | 平台专用的单点、看护、臂间串行与状态脚本（新；不在开源分支中） | 该平台既无 PBS 也无 Slurm，`afterany` 链没有对应物；而 `--strict-turns` 不许 fallback 轮，一次瞬时连接错误就终结整条 run，NSCC 的 grid 脚本已自带从 checkpoint 重试、Pod 侧没有 |
| Claude 订阅席位经 CLIProxyAPI 直连 | `hpc/nscc_claude_sub.pbs`、`nscc_claude_grid.pbs`（新） | 订阅席位此前只能经 OpenRouter 按量付费转发；该集群上 `platform.claude.com` 与 `api.anthropic.com` 其实直连可达，是唯一一条不需要 Worker 也不需要隧道的 provider 路线 |
| 登录节点上的**单一**共享 CLIProxyAPI | `hpc/nscc_cliproxy_shared.sh`（新） | 每作业各起一个实例时，多个实例各自刷新同一个 OAuth grant，被上游的 refresh-token 复用检测判成盗用、整个 grant 被吊销。一个 grant 只能有一个 refresher，这是并行跑的前提 |
| 跨站状态视图 | `hpc/status_all.sh`（新） | 每个平台一套 ad-hoc 单行命令，而且是被交一张臂/预算清单去循环——后加的臂不在清单里，静默死掉的点没人看见。改为自动发现 run 目录 |

两条与协议无关但会咬人的边界：`--provider claude` 在这条臂上**不可用**（anthropic SDK 不在 `hpc/requirements_fig2_nscc.txt` 里，那条路径从未被执行过），实际走的是 `--provider openai` + `llm/openai_compat.py`；出目录上只有 `nscc_claude_grid.pbs` 取 `flock`（`.nscc.lock`，节点上没有 `flock` 时降级成一条警告），而 `nscc_claude_sub.pbs` 与它共用同一个出目录却不取锁——两个作业指向同一预算点仍会同时打开同一个 `checkpoint.sqlite3`。两者都列进 §11 R7。

### 2026-08-26 追加：群名拼法的指针式报错

代码只动了 `construction/groups/catalog.py` 一个文件（回归 `tests/test_group_catalog_naming.py`），但其中一处会穿到 generator prompt，所以记在这里而不是只记在 01 §13。起因是同一个坑的两种拼法：文献（Mitten、Lin–Pryadko）按**群阶**给二面体群编号，本目录按**旋转阶**编号，照抄论文里的群名要么取不到群、要么取到另一个群。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| `S3` / `C{m}xS3` 报错时点名 `D3` / `C{m}xD3` | `construction/groups/catalog.py`（`_LITERATURE_RE`） | 文献拼法只换来一句 `unknown non-abelian group`，读者拿不到正确写法；而目录对 `C2xD{n}`→`D{2n}`、`C{m}:C2({k})`→`D{m}` 早就是"拒绝 + 指向规范名"，S3 是漏网的一类。没有加别名：`construction/lp.py:62` 把 `group.name` 回写进规范化 spec，一群两名会让同一个群以两种拼法进档案 |
| 阶上限报错追加二面体索引提示（`C4xD10` 点名 `C4xD5`） | `construction/groups/catalog.py`（`_dihedral_index_hint`） | `C4xD10` 语法合法却是**另一个群**（阶 80、无 GAP ID），此前只在阶上限处报 size 违规——静默语义错配伪装成预算问题。提示按 `KNOWN_GAP_IDS` 设闸（折半读数不是目录项就不出现），且只挂在上限被触发时——该依赖已于同日下一节把上限抬到 256 时兑现并补掉，`C4xD10` 改由保留名单无条件拒绝 |
| `catalog_summary()` 声明索引约定 | `construction/groups/catalog.py` → `construction/registry.py:213` → `funsearch.py:581` | 模型此前只能从报错里试错学到"论文的 S3 在这里叫 D3"。该串原样进 generator prompt，因此这同时是一次**不进协议快照**的 prompt 变更，见 §8 与 §11 R1 |


### 2026-09-01 追加：`bposd_structural_v3_2`（测量尺 + slow-loop 筛选机制）

两个互相放大的偏置一起修：**测得不准**（LSD 明显低估部分高 `k`、高距离的 BB/LP 结构）与**好结构留不下来**（跨 turn 状态只有一份按 plpr 排序的 runtime 名人堂，结构上突出但 plpr 不靠前的码没有保留、繁殖与终局槽位）。设计原则是：目标写进**机器选择规则**，不写进 prompt。

| 改动 | 文件 | 原来的失效方式 |
|---|---|---|
| BPOSD 单配置（`product_sum/50/osd_cs/10`）贯穿搜索、runtime、终局 | `config.py`、`qualification.py`、`cli.py`、`hpc/validate_fig2_campaign.py` | 仓库里同时存在三套 BPOSD 配置（默认 12/3、qualification 30/7、CLI `--osd-order 3`），且默认解码器集是 `(beam, lsd)`——"换了解码器"与"换了配置"都能悄悄发生 |
| 4-seed × X/Z 独立流的 `d_search`，逐 seed/sector 结果与 provenance 随行 | `verification/proxy_distance.py`、`verification/fast.py` | 单次 seed 的代理距离一旦获得独立繁殖槽，就足以把一个偶然虚高的码送上 max-d 端点；且 X/Z 共用一条 RNG，改一侧的 effort 会移动另一侧的随机流 |
| 三重资格拆分：`proxy_valid` / `runtime_screen_passed` / `structural_eligible` | `verification/fast.py`、`schemas.py`、`memory/archive.py` | `fast.valid` 同时承载"代理是否成功"、`d/√n` trust filter 与 runtime 资格。在 `n_data=288` 上 trust filter 拒 `d_search=4`——正是要保住的 high-k 端点——并且把完整 proxy 记录压成一条 error，候选从搜索里彻底消失 |
| `structural_proxy_elites`：cap 16 的 `(k, d_search)` proxy-Pareto 缓存，**在 MAP cell 竞争之前**更新，跨 turn 进 checkpoint | `ranking.py`、`memory/archive.py`、`workflow/state.py`、`workflow/graph.py` | MAP niche 只持久化每 cell 的 Q_eff incumbent，同 cell 落败者的 `spec/construct_source` 不再存在于任何地方；archive 每轮从种子重建，没有跨轮结构记忆 |
| slow 队列固定 4 槽：3 结构（high-k / high-d_search / crowding）+ 1 runtime，named promotion **占用**第 4 槽 | `campaign.py`、`orchestrator/{loop,plan}.py` | agent picks 作为 cap 之外的 EXTRA 槽追加，plan 能把队列撑大；且全 campaign 已测 hash 只在存在 `plan.promotions` 时才过滤，普通候选会跨 turn 重测 |
| 父代 = 跨 turn 8 structural + 4 runtime，2:1 交错覆盖 overlay；runtime 名人堂行补齐继承载荷并用 `construct_source+rng_seed` 重建校验 hash | `workflow/graph.py`、`evaluation/feedback.py`、`funsearch.py` | `hof_summary` 只存 `spec`，缺 `construct_source` 就根本渲染不成父代；`merge_parent_specs(slow_parents, original_parents)` 的 runtime-first 前缀让测量排名重新拥有了 lineage |
| 终局 3S+1R、fresh shots、**固定 T=12**、独立 RNG namespace | `workflow/graph.py`、`workflow/state.py` | 实际跑的是 `T=max(3, d_proxy)` 却标成 `T=d`；`d_proxy` 不是认证距离，既误标测量又让电路深度由搜索自己挑 |
| Generator 目标文本整体中和 + 三条回灌链切断 | `funsearch.py`、`evaluation/feedback.py`、`orchestrator/{manual,loop,plan,reflector}.py`、两份 prompt、`tools/*` | 只删两句不够：OBJECTIVE、"larger k wins"、archive 表的 `Q_eff higher is better` 与全局榜单序、`HOOK-DEGRADED … do not clone`、`prefer these over Q`、`best first`、`strongest verified`、父代行的比较分数、trust-filter 的 `or reduce n` 都在。更关键的是 **Policy 回灌**：`plan.directives`、reflector `next_turn_advice`、原始 generation notes 三条链都逐字进 Generator 的 S-B1，一个仍按旧规则思考的 Policy 可以把旧目标整段写回去 |
| `degraded_*` 降级为诊断标签；唯一硬排除是 `schedule_catastrophic` | `orchestrator/{loop,plan}.py`、`tools/acquisition_score.py` | hook 退化的代价已经反映在 BPOSD 实测 plpr 里，再按它隔离等于用 runtime 症状删除结构强码 |
| racing 冻结为关闭 + strict validator 逐层核对解码器/轮数/配额/`ldpc==2.4.1`/backend | `hpc/validate_fig2_campaign.py`、`workflow/checkpoint.py` | validator 只比对"调用方声明的 decoder 集"，于是 job script 能把自己的变体授权成合法协议 |

**测试**：`tests/test_structural_proxy.py`（d_search 协议 + reducer 不变量）、`tests/test_promotion_priority.py`（4 槽队列，全文重写）、`tests/test_structural_state_and_parents.py`（跨 turn 缓存、继承校验、父代配额、终局）、`tests/test_prompt_neutrality.py`（assembled prompt 无旧指令、三条回灌链封死）。

**未做（明确延期，见方案 §10）**：MILP incumbent/区间与 `epsilon_cert`、多套正式 frontier、SQLite registry、confidence sequence、SEC degradation 三态、全量 BP/OSD telemetry、Tier-2 denylist、多重比较校正、Beam 作全程解码器、在 prompt 里显式写结构目标。

### 2026-09-01 追加（二）：v3.2 实现审计的六项修复

首版 v3.2 落地后的一轮审计发现 2 个协议验收漏洞与 4 个会改变选码结果的跨轮问题，全部修复（258 测试全绿）：

| # | 修复 | 文件 | 原来的失效方式 |
|---|---|---|---|
| 1 | validator 强制 search 层实测 T=12：`slow_run_config.verification.rounds==12` + 每条 ok search/final ranking 行 `rounds` 逐行核对（缺字段即失败） | `hpc/validate_fig2_campaign.py` | 执行路径确实设了 T=12，但 validator 不核对——把测试产物的 search T 改成 5 仍然通过 |
| 2 | proxy 协议单一真源：fast/slow/pool 全部只读 `StructuralSearchConfig`（seeds/iters/algorithm 显式传入，算法声明不符即 fail-closed）；`proxy_iters` 降级为同步别名，非协议 effort 自动 retag `profile_id`（`+proxyN`）；validator 增查 run config 的 structural 块 + 每个结构 elite 的 `proxy_search` 实测 provenance（algorithm/seeds/iters/per_seed 表） | `config.py`、`verification/{proxy_distance,fast}.py`、`memory/archive.py`、`funsearch.py`、`evaluation/{pipeline,pool}.py`、`hpc/validate_fig2_campaign.py` | 冻结声明在 `StructuralSearchConfig`，实际 effort 却读 `VerificationConfig.proxy_iters`，slow 复算还吃默认 seeds——可以构造"声明 120、实跑 7"但仍标 v3.2 的配置 |
| 3 | 无损结构 ledger：`ArchiveMemory.structural_ledger`（hash-keyed、无 cap）承接全部 structurally eligible 行，`structural_proxy_elites` 变为惰性 batch reduce 的只读派生视图；ledger 以 `structural_ledger.jsonl` 落盘并经 `CampaignState.structural_ledger` 跨轮 | `memory/archive.py`、`funsearch.py`、`workflow/{graph,state}.py` | `reduce(已截成16的池 + 新候选)` 不满足结合律：LLM 输出顺序能改变最终结构精英 |
| 4 | 队列两个缺口：结构队列文件改为发射**整个有序池**（builder 做 hash/WL 去重后取满 3 槽）；named promotion 经跨轮 registry（= ledger）解析，不再只搜本轮队列 | `memory/archive.py`、`campaign.py` | 预截 3 条时一次 WL 碰撞把队列悄悄变成 2S+2R；Policy 合法点名上一轮候选时静默退回 runtime head |
| 5 | catastrophic 闭环：`ScheduleCatastrophicError` + 消息 token 跨进程识别，build/distance worker、pool `_assemble`、inprocess pipeline 全路径打 payload 级标记；`ranking_rows` 保证候选级终态行；`slow_queue_status` 把 catastrophic 计为"已识别并消费"（全 catastrophic 队列 = ok，不再挂掉 strict turn）；reconcile 兼收 failure_reason token；final shortlist 过滤 `unmeasurable_hashes` | `verification/circuit.py`、`evaluation/{worker,pool,pipeline,writeback}.py`、`workflow/graph.py` | 真实电路路径在 build 时 raise，flag 永远算不出来；全 catastrophic 的轮次被 strict audit 拒绝；catastrophic 码还能占终局槽 |
| 6 | prompt runtime facts 改稳定 hash 序（build 端 `best_results` + 渲染端双保险，`fact_order` 字段声明） | `evaluation/feedback.py`、`funsearch.py` | facts 列表沿用 plpr 排序——没有动词的有序列表本身就是隐式排序指令 |

新增测试：`tests/test_catastrophic_terminal.py`（终态语义全套）；`test_campaign_strict_turns.py` +6（T=12 逐行、structural 块、声明/实测 provenance、config 层 retag）；`test_structural_state_and_parents.py` +3（到达顺序无关、ledger 无损、全池发射）；`test_promotion_priority.py` +2（WL 碰撞、跨轮 registry）；`test_prompt_neutrality.py` +1（hash 序 facts）。

### 2026-09-01 追加（三）：第三轮审计的筛选修复（4 主 + 5 次）

| # | 修复 | 文件 | 原来的失效方式 |
|---|---|---|---|
| 1 | 端点只从最低 Pareto rank 取；tie-break 先比另一目标：`high_k=(-k,-d,n,-Q,hash)`、`high_d=(-d,-k,n,-Q,hash)` | `ranking.py` | k 并列时旧 §1.2 键先比 n_total，被 B=(20,10,n288) 严格支配的 A=(20,3,n100) 因为更小抢走 high-k 端点，D=(10,11) 掉出 3 槽 |
| 2 | crowding 按唯一 `(k,d)` 坐标计算、同坐标共享；坐标内部才按 `n_total↑/Q_search↓/hash`；截断键 = (boundary, -crowding, 坐标代表 hash, 坐标内 §1.2) | `ranking.py` | 逐行 crowding 把同一坐标的两个实现互相挤掉，`(10,8)` 的 n=200 在 tie-break 生效前击败 n=100 |
| 3 | final runtime 槽扫描整个 HoF 直到真正加入一个新 runtime 候选 | `workflow/graph.py` | 只试 `hof[:1]`：runtime 冠军已在结构槽时第四槽被结构池回填，产出 4S+0R |
| 4 | ledger 移出 checkpoint：run 根 `structural_ledger.jsonl` append-only store；state 只存 hash 游标 + reduced elites；canonical elites = reconcile 时**流式**两遍读 store 再 reduce；非支配排序改为坐标分组 + 2-D staircase 剥层（O(N + C log C)，剥到填满 cap 即停） | `workflow/{graph,state}.py`、`ranking.py`、`campaign.py` | 全量行进 state：28,800 候选上界下 checkpoint 数 GiB、内存 ~190 MiB、单次 reduce ~8.3 亿次两两比较 |
| a | 已测码的 WL 经 registry/store 流式恢复进队列去重 | `campaign.py` | hash 过滤挡不住 WL 等价的再发现，跨轮重测同构码 |
| b | validator 校验 final fresh seed：report `final_rng_base_seed` == namespace 派生值（≥1e12），final run config `seed` 精确相等，search run config `seed` < 1e12 | `hpc/validate_fig2_campaign.py`、`workflow/graph.py`（report 补记） | final seed 改成 0/123 也照样通过 |
| c | provenance 深检：per_seed 的 seed 序列必须等于声明 seeds、距离为正整数、聚合 d_x/d_z/d_search == 各最小值、available、行 d_search 与 provenance 一致 | `hpc/validate_fig2_campaign.py` | 只查表长度，伪造表可过 |
| d | `--strict-turns` + `--racing` 启动即拒（CLI + `build_protocol_snapshot` 双拦截，检查请求标志而非 worker 调整后的有效值） | `cli.py`、`workflow/state.py` | 跑完 60 轮才被外部 validator 拒绝 |
| e | docs/04 §9 正式命令改为 v3.2（无 racing/bposd 默认）；三份 FIG2 README 顶部加"已退役协议、必须绑定旧代码快照"警示 | `docs/04_architecture.md`、`hpc/` 下三份平台 README（不在开源分支中） | 文档仍把 LSD+racing 写成现行可合并协议 |

新增测试（268 全绿）：端点支配（审计 A/B/C/D 样例）、坐标共享 crowding + n 优先、final 槽跳过重复、store 流式 reduce 顺序无关 + crash 去重、WL 等价再发现过滤、final seed 三向拒绝、伪造 provenance 三向拒绝、strict+racing 快照拒绝（含 workers=1 掩蔽路径）、validator/graph seed 公式互钉。另有 200 随机池的分层性质校验（快速剥层 == 支配公理）。

### 2026-09-01 追加（四）：`bposd_structural_v3_3`（解码并行 + 超时反偏置）

v3.3 **不改** v3.2 的科学尺：BPOSD 仍是 `product_sum / 50 / osd_cs / 10`，4-seed `d_search`、结构 Pareto 保留、3S+1R slow/final、8S:4R 父代与固定 T=12 均不变。这次升级修的是执行几何与超时选择偏置：v3.2 默认 `shot_shards=1`，即使 PBS 申请 128 CPU，4 候选 × X/Z 也只有 8 个解码作业；同时单一 timeout 可以把难码的部分基/shard 变成一个仍可排名的部分测量，这会让计算难度变成隐藏的码质量筛选器。

| 改动 | 现行语义 | 防止的失效 |
|---|---|---|
| 16-way fixed shot sharding | search/final 同用 `shot_shards=16`；4 候选 × 2 基 × 16 = 128 个单核 decode work units | 128 核整节点长期只有 8 个解码任务，CPU 平均停在 3–5% |
| typed watchdog | build 3600s；decode shard 1800s；distance 1800s | 一个单体 timeout 既过早杀大码，又无法分辨哪个阶段卡住 |
| immutable timeout replay | `timeout_retries=2`，只重放同一 request/seed，最多共 3 次 | 超时后重新生成候选，直接用易码替换难码 |
| fixed-shot completeness gate | 任一必需 X/Z shard 不全则候选与队列都是 `measurement_blocked`；部分计数只作诊断 | 一个便宜候选的完整结果掩盖另一个大码的超时，并让 strict turn 错误地通过 |
| validator 执行几何深检 | snapshot、search/final run config 与 candidate `job_config` 均核 `shot_shards` 及四个作业策略字段 | launcher 只在命令行自称已分片，但产物实际仍按单 shard 跑 |

`shot_shards` 是协议字段，不是可随意调的性能开关：它改变 shard-index seed 和采样流，所以 profile 升为 `bposd_structural_v3_3`，strict 模式只接受 16，v3.2 不可 resume 到 v3.3。NSCC 活动 launcher 是 `nscc_claude_grid_v33.pbs`，提交与监控分别用 `submit_claude_v33_grid.sh` 和 `claude_v33_status.sh`；旧 v3.2/legacy launcher 不得在当前代码下声称 v3.3 产物。

**已知边界**：当前重试在一次 `run-campaign` 进程内进行，candidate artifact 显式记 `persistent_shard_ledger=false`。尚无跨 PBS 作业持久化的逐-shard 完成账本；如果同一 shard 在三次尝试后仍超时，v3.3 Claude launcher 保留 artifact/checkpoint 后停止该点，由人诊断，不会返回 Generator 换码。这个边界记入 §11 R20。

### 2026-09-02 追加（五）：`bposd_structural_v3_4`（按 shot 预算的 watchdog、终局 128 shards、decode_infeasible 候选级终态）

v3.4 **不改** v3.2/v3.3 的科学尺：BPOSD 仍是 `product_sum / 50 / osd_cs / 10`，4-seed `d_search`、结构 Pareto 保留、3S+1R slow/final、8S:4R 父代、固定 T=12、racing 关闭、search 每基 16 shards 与 fixed-shot 完整性门均不变。这次升级修的是 v3.3 的执行几何在实机上**跑不完**、并把"慢"误判成"坏"的问题。NSCC 2026-09-02 b360 canary 实测：

- BPOSD 在 n_data 144–180 的典型 bb 码上 **0.65–0.94 s/shot**；`[[180,2,20]]` lp_abelian **5.2–5.5 s/shot**（BP 不收敛 → 几乎每个 shot 都走 OSD）。
- v3.3 冻结的 flat 1800 s decode-shard watchdog + 16 shards：search shard = 1250 shots，慢码需 6,500–6,800 s；final shard 在 b260–360 = 12,500 shots，需 8,000–68,000 s → 正式终局阶段在 b220+ **永远不可能完成**，且一个慢码就让整个点 `FATAL_STOP`。
- 确定性超时的同请求重放 ×3 白烧了 32 shards × 2 × 1800 s。
- 两个已确认的管道 bug：(BUG-1) worker `dem_load_error` 被持久化为 shard "timeout"，无 `job_failures` 记录、不重试（`pool.py` 结果循环与 `sampling.merge_shard_results`）；(BUG-2) 被杀 worker（OOM SIGKILL）变成 `subprocess_error`，returncode 丢失、从不重试。

| 改动 | 现行语义 | 防止的失效 |
|---|---|---|
| 按 shot 预算的 shard watchdog | 新协议字段 `decode_shot_budget_s=8.0`；有效 watchdog = `max(decode_shard_timeout_s=1800, 8 × 该 shard 的 shots)`：search 1250 shots → 10,000 s，final 1562 shots → 12,500 s；`decode_shard_timeout_s` 降为 **floor**。`pool.effective_decode_shard_timeout_s(verification, shots)` 是纯函数，写入 candidate `job_config.effective_decode_shard_timeout_s`；budget ≤ 0 退回 floor，结果 ≤ 0 即关闭 | flat 1800 s 在大 shard 上必然超时，把计算成本当成码质量筛选器 |
| 终局 128 shards | 快照新增 **`final_shot_shards=128`**（`shot_shards=16` 仍只指 search），`final_slow_node` 读它；strict 模式要求 16/128，非 strict 值给 `protocol_profile` 追加 `+shards{N}` / `+finalshards{M}`；校验 `1 ≤ final_shot_shards ≤ final_shots`；CLI `--final-shot-shards`（默认 128） | 12,500-shot 的终局 shard 需 8,000–68,000 s |
| 超时不重放 | `timeout_retries` 默认由 2 改 **0**（strict/validator 冻结）；语义不变，只是不再重放确定性 watchdog 超时 | 同一确定性超时 ×3 重放白烧 shard 时间 |
| 瞬态错误重放一次 | 新字段 **`transient_retries=1`**：对 `subprocess_error`（无可解析 JSON / 被杀）与 worker 状态 `dem_load_error` 做同请求重放；**不适用于 timeout**。`job_execution` 记 attempts / retries_used / retry_limit / transient_retry_limit / transient_retries_used / timeout_s / **returncode**（`retries_used` 只计 timeout 重放、与 `retry_limit` 配对；瞬态重放计入 `transient_retries_used`；`attempts` 是总次数，timeout 的 failure_reason 引用的 `on all <n> attempts` 也是总次数），`timed_out=true` 只在最终超时出现 | BUG-1 / BUG-2：真实故障被记成 timeout 或被静默吞掉 |
| shard cell 记真实状态 | 非 ok 作业按 outcome status（timeout / subprocess_error / dem_load_error / 其它）为每个 (decoder, basis) 合成失败 cell（shots 0、failure_reason、job_execution）；每个非 ok 的 shard/build/distance 结果都追加 `job_failures`；`merge_shard_results` 增 `shard_runtime_s` 与 `shard_attempts`；None 槽只在作业从未返回时保留 | dem_load_error 伪装成 timeout，事后无法诊断 |
| `decode_infeasible` 候选级终态 | 结果级：所有必需基都在、且必需基上所有非 ok shard 状态**恰好全是** `timeout` → `status=decode_infeasible`，否则仍 `measurement_blocked`（二者都 `measurement_complete=false`、`combined` 置空且 `failure_reason=fixed_shot_measurement_incomplete`、`partial_combined` 保留）。候选级优先序：schedule_catastrophic > build 错误/超时 > 全 ok > 任一 measurement_blocked > 任一 decode_infeasible（`payload.decode_infeasible=true`，`failure_reason="decode exceeded the frozen per-shot budget: <basis>:<n_timeout>/<n_shards> shards (timeout_s=<t>)…"`）> no_valid_results。`count_incomplete_candidates` 排除它；summary 增 `decode_infeasible_candidate_count`。该判定只存在于 job-pool 路径（`workers > 1` → `evaluate_queue_with_jobs`；正式跑 127 workers 必走此路径），串行路径 `workers=1`（`pipeline.evaluate_slow_spec`）未改。所有入围候选都 decode_infeasible 时终局 shortlist 为空：非 strict 以 `final_slow_status="empty_shortlist"` 正常结束（本机强制 budget=0.001 s/shot 的 canary 实测），strict 仍按原逻辑报错 | 一个确实超预算的码让整个点 `FATAL_STOP`，后面的 turn 全部作废 |
| 队列与 workflow 语义 | `slow_queue_status(decode_infeasible_candidate_count=…)`：`effective = record_count − catastrophic − decode_infeasible`，effective ≤ 0 即 `ok`；`measurement_blocked` 触发条件不变。ranking 行 status=`decode_infeasible`、plpr None。`unmeasurable_hashes` 收它（不入 final shortlist、不重排队，结构账本行保留——其 k/d_search 仍是合法父代）；`_audit_final_measurements` 与 validator 视为终态跳过，final 仍需 ≥ 1 个 ok 行 | 队列因一个慢码被判 `measurement_blocked`，或慢码被反复重排队 |
| validator 执行几何 | `PROTOCOL_PROFILE=bposd_structural_v3_4`；snapshot `shot_shards==16` 且 `final_shot_shards==128`；每 turn search run config / candidate `job_config` shot_shards==16，final run config / final candidate `job_config` shot_shards==128；`FORMAL_JOB_POLICY` = {build 3600, decode_shard 1800, distance 1800, timeout_retries 0, decode_shot_budget_s 8.0, transient_retries 1}；拒绝 v3.3 值（timeout_retries 2 / 无 budget / final shards 16） | launcher 自称 v3.4 但产物仍按 v3.3 几何跑 |
| launcher / ops | `hpc/nscc_claude_grid_v34.pbs`（job `omni_claude_v34`、`_v34` 目录、常量 `SHOT_SHARDS=16 / FINAL_SHOT_SHARDS=128 / DECODE_SHOT_BUDGET_S=8 / TIMEOUT_RETRIES=0 / TRANSIENT_RETRIES=1`、freshness 检查 import `FORMAL_FINAL_SHOT_SHARDS`）、`submit_claude_v34_grid.sh`、`claude_v34_status.sh`（shards "search 16 / final 128"、逐 run 目录汇总 `decode_infeasible`）；`measurement_blocked` 仍 `FATAL_STOP`，**`decode_infeasible` 不停止**；`claude_v33_diagnose.py`（版本无关，名字不改）识别 decode_infeasible 并显示 `shard_runtime_s` max/mean | 与 v3.3 相同的安全机制（共享 CLIProxyAPI 单 refresher、配额轮询、strict 重试、flock、walltime reserve、冻结 slow_workers 检查）全部保留 |

`final_shot_shards` 与 `decode_shot_budget_s / transient_retries` 都是协议字段，`timeout_retries` 默认也变了，因此快照与 v3.3 必然不同：**v3.3 checkpoint 一律不能用 v3.4 `--resume`**，launcher 改用 `_v34` 目录。v3.3 三件 launcher/提交/状态脚本保留为历史，不得在当前代码下声称 v3.4 产物。可比性：v3.3 → v3.4 的 BPOSD/结构选择尺不变、search RNG 流不变，但终局 shard 数改变了终局采样流，且超预算码由"停点"变成"终态退出"——两版结果分栏标注，不混池（R18 同理）。

**上机顺序**（当时的账号本地手册，不在开源分支中）：先跑**不带 `FINAL_SHOTS_OVERRIDE`** 的 b360 单轮 canary，让 200k-shot × 128-shard 的终局阶段真实完成（预期约 3.5 h），核对峰值内存 < 80 GB、终局每基 128 个 ok shard、`shard_runtime_s` 与 watchdog 的余量、`decode_infeasible` 只出现在确实 > 8 s/shot 的码上；然后 `submit_claude_v34_grid.sh` 放全网格。

**v3.4.1（2026-09-02 审查后收紧）**：

- `decode_infeasible` 只承认**纯首次尝试的 watchdog 超时**：瞬态重放（`transient_retries`）之后再超时的 shard 记为 `timeout_after_transient`，归入 `measurement_blocked`（仍停点），不再被当成"确实超预算"的终态。
- launcher（`nscc_claude_grid_v34.pbs`）对"终局全部 decode_infeasible"（strict 终局审计报 `no successful final measurement rows`，会被 `looks_like_strict_abort` 匹配）立即写 `FATAL_STOP` reason=`final_all_infeasible` 并退出，不再把整个确定性终局重放 STRICT_RETRY_MAX 次；`FINAL_SHOTS_OVERRIDE < FINAL_SHOT_SHARDS` 在启动时以 exit 2 拒绝；header 列出全部 FATAL_STOP reason。
- prompt 侧 `slow_feedback.best_results` 只保留 `status == "ok"` 的行（仍按 code hash 稳定排序，`fact_order` 不变），非 ok 行（plpr None）不再以 `per_logical_per_round=unknown` 出现在 generator prompt 里；`state.py` 关于 shard 数的注释更正为：`shard_seed` 不依赖 shard 数，改变的是每 shard 的 shots 划分（采样流），resume 仍因快照键不同而拒绝。

### 2026-09-02 追加（六）：`bposd_structural_v3_5` —— 噪声模型回归论文 SI C（去 idle 噪声），SEC 仍为 LRC（同日随后由追加（七）改为 `twobasis_nonint_v1`）

用户决定（2026-09-02）：「SEC 这么做没有问题（LRC 保留），但是引入的 noise 这块还是按照我组员的论文里面来，这个 idle 噪声不要。」组员论文 SI C b "Noise model" 原文：

> All circuits use a uniform circuit-level depolarizing model controlled by a single physical error rate p. Each qubit reset is followed by single-qubit depolarizing noise of strength p; each controlled-NOT is followed by two-qubit depolarizing noise of strength p; each ancilla measurement is preceded by a bit-flip error of probability p; and the final data readout is preceded by a basis-conjugate flip of probability p.

没有 idle 噪声；p = 0.005。v3.5 **不改** v3.2–v3.4 的科学尺与执行几何，只改正式协议的噪声模型；SEC 仍为 LRC（本节写于 SEC 切换之前，SEC 的最终形态见追加（七））。本节按 v3.5 contract 记录落点，权威顺序仍是代码 > 本文。

**精确 lowering**（`verification/noise.py` 新增 `NoiseProfile.UNIFORM_PAPER = "uniform_paper_v1"`；`sec/stim_builder.py::build_lrc_memory_circuit` 新增 `kind == "uniform_paper"`，LRC 分段 / 调度 / detector / observable 与 legacy uniform 完全一致，只有噪声指令不同；detector / observable / 测量计数与同一 schedule 的 legacy 发射逐项相等）：

- 每个 reset（全局数据制备 `R`/`RX` **以及**每个 ancilla 的 `R`/`RX`）：先发 reset，再对同一批目标发 `DEPOLARIZE1(p)`；
- 每一层 `CX`：`CX` 后紧跟 `DEPOLARIZE2(p)`；
- 每个 ancilla 测量：`M` 前紧跟 `X_ERROR(p)`，`MX` 前紧跟 `Z_ERROR(p)`（与 legacy 相同）；
- 终段：推迟的 `M(zanc)` / `MX(xanc)` 按上一条加噪；之后**不再**对数据发 `DEPOLARIZE1`；终局数据读出前发基共轭翻转——X-memory 为 `MX(data)` 前 `Z_ERROR(p)`，Z-memory 为 `M(data)` 前 `X_ERROR(p)`；
- **任何位置都没有 idle `DEPOLARIZE1`**（phase-2 的 idle 标注对该 kind 不发射）；TICK 结构不变；
- `saturate_noise=True`（结构审计）仍走 legacy uniform lowering，与非 legacy profile 组合仍 `ValueError`；`noise_point=None` 仍表示 legacy 发射（golden 对拍、审计路径逐字节不变）。

**为什么**（本机 2026-09-02 对 Mitten [[200,40,12]] 与 BB144 的诊断）：

- 噪声体量（本机 2026-09-02 计数，R=12）：LRC legacy 发射对 Mitten [[200,40,12]]（GAP(40,5)，w=9）产生 **20,248** 个单比特 `DEPOLARIZE1` 目标（idle + 终段数据），论文模型只有 **2,120**（仅 reset 之后）——单比特噪声暴露 **9.6×**，总噪声事件 1.45×；BB144（w=6）只有 1.8× / 1.05×。idle 噪声因此是一个**随 check 权重与 LRC 深度增长、按 family 不均匀**的额外惩罚，系统性压低高权重 LP/非阿贝尔码。
- 测量（本机 BPOSD 50/10，T=12）：Mitten 码在 legacy（LRC + idle）下 p=0.005 两基 LER **= 1.0**（2600/2600），p=0.002 每基 0.78–0.80；同一 LRC 电路去掉全部 idle `DEPOLARIZE1` 后 p=0.002 降到 0.18–0.19，但 p=0.005 仍 0.97–0.98。按论文 SI C 协议（论文自身的**单边顺序** SEC + 无 idle）p=0.005 得 LER_X=0.0019 / LER_Z=0.0038、plpr≈1.2e-5——论文数量级可复现。
- 结论：idle 项是 v3.5 要去掉的、可归因且按 family 偏置的部分；去 idle 后 LRC 与论文单边 SEC 之间在 p=0.005 仍有差距，这是用户明确保留 LRC 的已知代价，不在 v3.5 范围内。LRC 的 hook 结构（Δmin / τA / `d_circ_ext` 排名、SBB Thm 1 的 residual 集）保持不变：Mitten 码 `d_circ_ext(x,z)=(21,16)`、无 `delta_one` 淘汰，不是调度问题。

| 改动 | 现行语义 | 防止的失效 |
|---|---|---|
| 噪声 profile `uniform_paper_v1` | 见上表（§6）与 lowering；`NoiseProfile.UNIFORM_PAPER`，docstring 注明 OmniQEC paper SI C b、无 idle | 逐 tick idle 噪声按 check 权重 / LRC 深度、按 family 不均匀地惩罚高权重码 |
| 默认与 plumbing | `NoiseSweepConfig.profile` 默认 UNIFORM_PAPER；`uniform(values)` = paper（source_kind `paper_protocol_parameter`、source_reference "OmniQEC paper SI C uniform v1"）；新增 `uniform_legacy(values)` 返回旧 profile 与旧字符串；`noise_sweep_from_legacy_p_values` → paper（`--p/--p-values` 别名 = 当前 uniform 协议）；`UNIFORM_PROFILES` frozenset 取代所有 `== UNIFORM_LEGACY` 的"是否 uniform 族"判断（config / cli / stim_builder / circuit / sampling / progress csv）；CLI `--noise-profile` 默认 `uniform_paper_v1`，`--p` 别名对两个 uniform profile 都合法，SI/Helios 混用错误不变；`RACING_ANCHOR_P[UNIFORM_PAPER] = 0.005`；`slope_offset=0.958` 两个 uniform 都保留 | 遗留的 `--p` 路径静默落回 idle 模型；某处 `== UNIFORM_LEGACY` 把 paper profile 当成非 uniform |
| 协议身份与冻结 | `StructuralSearchConfig.profile_id = "bposd_structural_v3_5"`，所有 v3_4 文案改 v3_5；快照新键 `noise_profile`（str）/ `noise_values`（list[float]）；strict 要求 `uniform_paper_v1` / `[0.005]`（否则 ValueError，与 shards 检查同风格）；非 strict 的非 paper profile 给 `protocol_profile` 追加 `+noise<profile>`；resume 逐键比对，v3.4 快照缺键即按协议不匹配拒绝 | 一个 v3.4 checkpoint 被静默续成 v3.5 |
| validator | `PROTOCOL_PROFILE = "bposd_structural_v3_5"`、`FORMAL_NOISE_PROFILE = "uniform_paper_v1"`、`FORMAL_P = 0.005`；核快照两键与每条 slow candidate/result 的 `noise_point.profile` / value | launcher 自称 v3.5 但产物在别的噪声下测 |
| launcher / ops | `hpc/nscc_claude_grid_v35.pbs`（job `omni_claude_v35`、`_v35` 目录、`readonly NOISE_PROFILE="uniform_paper_v1"` / `P_VALUE="0.005"`，每次 run-campaign 显式 `--noise-profile "$NOISE_PROFILE" --p "$P_VALUE"`，启动自检 `V35_DEFAULTS_OK` 断言 `NoiseSweepConfig().profile == uniform_paper_v1`、`profile_id == bposd_structural_v3_5`、validator 常量与 v3.4 几何常量）、`submit_claude_v35_grid.sh`（formal gate `RUN_TAG=v35`）、`claude_v35_status.sh`（显示快照 `noise_profile`，非 paper 值打 `⚠noise=`）；v3.4 三件头部加 HISTORICAL 行；所有 v3.4 decode-budget 常量与 FATAL_STOP 分支原样保留 | 部署陈旧时在 `_v35` 路径下冻结 v3.4 / idle 噪声 |

**没有改的**：SEC 调度合成/排名（LRC、色置换枚举、Δ/τA 三段淘汰）、解码器配置（BPOSD `product_sum / 50 / osd_cs / 10`）、shards / watchdog / 重试（search 16 / final 128、8 s/shot、3600/1800/1800、0/1）、plpr 公式（A4–A5）、prompt、种子、family、结构保留通道、3S+1R / 8S:4R、固定 T=12、racing 关闭、`slope_offset=0.958`（fast-loop 代理校准，两个 uniform profile 都保留）。

**可比性**：v3.4 → v3.5 的 BPOSD / 结构选择尺、执行几何与 RNG 流不变，但噪声模型变了，所有 LER/plpr 绝对值都会移动——**v3.4 产物与 v3.5 不可合池**，同一张 Fig.2 只能全部用 v3.5 重跑或显式分栏（R21，R18 同理）。v3.4 的 s/shot 实测是 idle 噪声下的数字，只作 v3.5 的上界。

**上机顺序**（当时的账号本地手册，不在开源分支中）：等 NSCC 上的 v3.4 canary 结束（**campaign 运行期间禁止 rsync**，worker 子进程在 spawn 时才 import 代码）→ rsync → pytest / `V35_DEFAULTS_OK` → `POINTS=360 TURNS=1 RUN_TAG=v35_canary VALIDATE_FORMAL=0 CHAIN_LINKS=2` 的 b360 单轮 canary（核快照 `noise_profile` / `noise_values`、candidate `noise_point`、终局 128 ok shard、重测 s/shot 与内存）→ `submit_claude_v35_grid.sh` 放全网格。

**测试**：`tests/test_v35_noise_profile.py`（lowering 保真：paper profile 的 `DEPOLARIZE1` 只紧跟 reset 且目标数等于 reset 目标数、末测量层后零 `DEPOLARIZE1`、每个 CX 后 `DEPOLARIZE2`、ancilla M/MX 前匹配翻转、终局读出前基共轭翻转、detector/observable/测量计数与 legacy 相等；默认值；CLI 默认；快照冻结 / strict 拒绝 / resume 不匹配；validator 常量；`RACING_ANCHOR_P`）、`tests/test_hpc_v35_scripts.py`（13 条：v35 三件、噪声旗标、`V35_DEFAULTS_OK`、FATAL_STOP 分支、v34 HISTORICAL、手册 / README / 本文引用原文）；golden SEC 测试不变（`noise_point=None` ⇒ legacy）；所有钉在 `bposd_structural_v3_4` 的旧断言改 v3_5。

### 2026-09-02 追加（七）：v3.5 SEC = `twobasis_nonint_v1` —— 两基不交错 coloration 电路 + 基内层序认证

用户决定（2026-09-02，原话）：「单一的基肯定是不对的 … 后续 helios 需要做 bias … 必然需要两个基」→ **两基每轮都测**；「不做任何的交错，两个基完全独立开」→ **不交错**（每轮 = X 半轮 [reset X ancilla、X 层、测] 然后 Z 半轮）；「基内部可能还是需要 scheduling 来减小 hook 的影响，但开销会小很多」→ **廉价的基内层序搜索 + 认证**；「杜绝一切可能造成大码距或者大 k 被误杀的情况」→ 调度规则跨 family 统一，每个候选序都记录认证证据，认证结果**只作证据、从不作停止**。Detector 两基（「子决定是显然的，依然是两基」）；噪声沿用追加（六）的 `uniform_paper_v1`；非交错深度 w_X + w_Z + 4 被接受（idle 在各平台都是次要因素，DD 论证）。权威顺序仍是代码 > 本文；新增/改动文件：`sec/layers.py`、`sec/twobasis_rank.py`、`sec/twobasis_builder.py`（新），`sec/stim_builder.py`（拆为共享发射器）、`sec/coloring.py`（只增）、`sec/audits.py`、`construction/{bb,lp,registry,build}.py`、`schemas.py`、`config.py`、`verification/circuit.py`、`evaluation/{common,worker,pipeline,evidence}.py`、`qualification.py`、`export_stim.py`、`cli.py`、`workflow/state.py`、`hpc/validate_fig2_campaign.py`、v35 launcher 三件。

**为什么**（本机 2026-09-02，BPOSD 50/10，T=12，p=0.5%，两者都**无 idle**）：

| 码 | LRC（`lrc_ranked_v1`）LER_X / LER_Z | 论文 SI C 的单边顺序 SEC LER_X / LER_Z |
|---|---|---|
| Mitten [[200,40,12]]（GAP(40,5)，w=9） | 0.98 / 0.97 | 0.002 / 0.004 |
| BB144（w=6） | ≈0.07 | 0 / 1040 shots（未观测到逻辑错误） |

去掉 idle 之后 LRC 与论文 SEC 的差距仍在两个数量级以上，且随 check 权重加剧——这是 LRC **时间线本身**（左右两半交错、hook 传播集依赖交错细节）的代价，不是噪声体量问题，因此 SEC 换为不交错的两基 coloration 电路；LRC 代码与 golden 对拍（`tests/test_sec_golden.py`）原样保留，策略 id `lrc_ranked_v1` 仍可显式选用，但与 twobasis 产物**不可合池**。

**时间线**（`sec/twobasis_builder.py::build_twobasis_memory_circuit(hx, hz, schedule, *, basis, rounds, p, noise_point=None, saturate_noise=False)`，layout 与 LRC 相同：data 0..n−1 | zanc n..n+m_Z−1 | xanc 其后；`rounds ≥ 1`，不再要求 ≥ 2）：

```text
S0   RX(data)  [X-memory] / R(data) [Z-memory]      ← 全局只制备数据；ancilla 在每轮内 reset
每轮 r = 1..T:
  A    RX(xanc)                                         1 tick
  X_i  CX(xanc→data)，X 层 i = 1..|LX|                  每层 1 tick（层 = matching）
  B    MX(xanc)  → 记录 "xcycle"                        1 tick
  C    R(zanc)                                          1 tick
  Z_j  CX(data→zanc)，Z 层 j = 1..|LZ|                  每层 1 tick
  D    M(zanc)   → 记录 "zcycle"                        1 tick；r = T 时同一段接终局
终局  X-memory: MX(data) / Z-memory: M(data)            重构 detector 只对 W 基
```

每轮恰 |LX|+|LZ|+4 个 tick（uniform 族 `circuit.num_ticks == T·(|LX|+|LZ|+4)`），测量数 T·(m_x+m_z)+n；`TwoBasisSchedule.depth = |LX|+|LZ|+4`（BB 16、mitten 22；LRC 为 w+2 量级）。X ancilla 与 Z ancilla 之间**没有任何 CX**（静态审计的门向检查把任何触碰两类 ancilla 的 CX 计为 `gate_orientation_errors`）。

**层**（`sec/layers.py::build_layer_sets(hx, hz, *, layer_blocks)` → `{"X": [...], "Z": [...]}`，Edge = (check, 全局数据列)）：

- 结构路径（bb / lp_abelian / lp_nonabelian）：`CompiledCode.layer_blocks`（新字段，默认 None；schema `layer_blocks_v1`，`block_size` g = bb 的 ℓ·m / lp 的 |G|；每项 `[cb, db, perm]` 断言 `H_W[cb·g+perm[t], db·g+t] = 1`，即 `_lift` / `np.kron` 的每个环项 = 一个 check 块与一个数据块之间的置换矩阵；`build.py` 重建两基核对，不进 `code_hash` / `compiled_code_metadata`）。以 check 块 ∪ 数据块为顶点、每项为一条边建二分**多重**图，用新的 `sec/coloring.py::color_bipartite_multigraph`（边索引版 König/Kempe，`minimal_edge_coloring` 字节不变）最小边着色；每个颜色类 = 一层 = 若干互不共享 check 块与数据块的原子 matching 之并。层数 = max(H_W 最大行权, 最大列权)：BB72 / BB144 = w = 6，D6 lp = 8、C3×C3 lp = 7（都等于 w）；列度大于行度的 LP 形状合法地多于 w 层，测试断言的是最大度公式而不是字面 w。
- 回退路径（hgp，或任何 `layer_blocks=None` 的码，含 `verification/structural.py` 的裸矩阵路径）：`minimal_edge_coloring(h)` 不变。
- 两条路径都断言：每层是 matching（check 与数据列都不重复），所有层的边多重集 = H_W 的非零恰各一次。

**Detector 规则**（两基，由 `_Timeline(detect_kinds={"X","Z"}, first_event_kinds={W})` 驱动共享发射器）：记忆基 W 的 check 第 1 轮发单项 detector，之后相邻两轮配对；反基 check 第 1 轮**不发** detector（随机值，只记录），第 2 轮起相邻配对；终局只对 W 的 check 发重构 detector（数据读出奇偶 vs 最后一次 W ancilla 结果）；observable = `css_logicals` 的 W 逻辑。计数 = T·m_W + (T−1)·m_opp + m_W，observables = k。LRC 用 `detect_kinds = first_event_kinds = {W}` 走同一发射器，输出**逐字节不变**（`tests/test_sec_golden.py` + 新对拍：BB72 与 D6 lp_nonabelian、两基、rounds=3、四种 lowering 与 `saturate_noise`，`str(new) == str(old)`，old 取自改动前快照）。

**噪声 lowering**：同一 `_emit_circuit`（`_resolve_lowering` 给出 kind ∈ uniform / uniform_paper / si1000 / helios）。`uniform_paper`：每个 reset（含 S0 与每轮两类 ancilla）后 `DEPOLARIZE1(p)`、每层 CX 后 `DEPOLARIZE2(p)`、`M` 前 `X_ERROR(p)` / `MX` 前 `Z_ERROR(p)`（含终局数据读出的基共轭翻转）、无 idle、无终局 `DEPOLARIZE1`——正式协议。`uniform_legacy`：逐 tick idle（twobasis 的 idle 窗口是整个数据集 `(range(n),)`），只用于 golden / `saturate_noise` 结构审计。`si1000` / `helios` 在新时间线上**只保证正确性**（X-memory 第 1 轮 prep-H 与 frame-H 相邻、不合并，无 window hoisting）：能构建、detector 确定（smoke），深度 / idle 记账未调优，**不得作为硬件阶段结果**。

**基内层序搜索 + 认证**（`sec/twobasis_rank.py::rank_twobasis_schedules(hx, hz, *, layer_blocks, policy, d_code_proxy)`；协议常量 `config.SecPolicyConfig`：`sec_policy_id="twobasis_nonint_v1"`、`sec_candidates=16`、`sec_cert_iters=400`、`sec_cert_seed=0`，`DEFAULT_SEC_POLICY`；LRC 对应 `LRC_SEC_POLICY`（64 / 100）：

- 候选：每基独立，i=0 = canonical（颜色序），i=1..15 = `counter_permutation(len(L), candidate=i, slot=0(X)/1(Z))` 置换层序。
- 认证：**不交错 ⇒ SBB Thm 1 精确成立**（电路距离 = 扩展码距离），不需要编译电路。对每个 check 按层序排其边，residual 后缀按 `schedule_rank.residual_suffixes` 语义（后缀权 w−1..1），`extended_matrices(h_detect, logical_rows, residuals)` + `kernel_basis` + `_proxy_side`（随机化 ISD，每侧 400 iters，seed = `sec_cert_seed + i·7 + side`，side 0 = Z-kind、1 = X-kind，与 `d_ext_probe` 同映射）→ `d_ext` 估计；`delta_one_residuals` 命中（某 residual 模 stabilizer 等于逻辑）→ 该候选淘汰。
- **ISD 估计是上界**（找到的最轻扩展码字），只作证据：`d_ext_x/z`、`chosen_index_x/z`、`per_candidate`、`schedule_degraded_x/z = d_ext < d_code_proxy` 全部写入 `RankedSchedule.metadata`、`sec_ext_evidence`（新增 `sec_policy_id`）与每条 candidate 的 `memory_experiment.schedule`，**从不作为停止**；catastrophic（某基全部 16 个候选 delta-one）→ `ScheduleCatastrophicError`，与 LRC 相同。
- 选择：每基独立取最大 `d_ext`（None 最低）、平手取最小序号（canonical 胜）；两次调用结果相等。
- 成本（本机实测）：BB72 每侧 0.61 s（32 次认证 ≈ 20 s，LRC `rank_schedules` 2.65 s）；BB144 每侧 2.4 s（≈ 77 s vs 10.3 s）；n=288 / w=9 一类每候选可达数分钟——受 `build_timeout_s=3600` 约束、由 synth cache 摊销；cache key 含 policy id / K / iters / seed / `layer_blocks` 哈希（LRC 支仍哈希 `sector_sizes`），旧 `synth_*.pkl` 全部一次性失效。观测：BB144 X 侧 canonical 序 `d_ext` 探到 9，LRC 优胜序 11–12——层序选择确实有影响，K=16 **不能悄悄缩小**。

**深度代价**：`uniform_paper_v1` 无 idle 项，代价只出现在 si1000 / helios 的 idle 上：每个数据比特每轮额外 idle 去极化量级 ≈ (w+2)·p/10（contract 估算；精确 idle tick 数 = |LX|+|LZ|+4 − deg_X(q) − deg_Z(q)，BB 为 w+4，且 reset/measure tick 上 si1000 的 idle 是 2p 档，尚未调优）。

**什么没有被声称**：hook 错误**没有被消除**——非交错只保证 ancilla 故障的传播集恰是"同一 check 的层序后缀"（因此可精确认证），基内层序只在这些后缀之间选最优；`d_circ_ext` 数值与 v3.4 LRC 的 `d_circ_ext` **不可比**（residual 集不同、100 vs 400 iters、上界更紧），`degraded_ext` 行跨策略不可比，evidence blob 里的 `sec_policy_id` 就是为此；`sector_h` / 非平凡 `h_mask` 在两条策略下仍 `NotImplementedError`（R16），`layer_blocks` 的块索引不感知 mask，若日后接入须与 `audit_twobasis_static` 的门向检查一起改。

| 改动 | 现行语义 | 防止的失效 |
|---|---|---|
| 策略开关 | `verification/circuit.py::build_memory_experiment_pair(..., sec_policy=None, layer_blocks=None)`；`policy = sec_policy or DEFAULT_SEC_POLICY`；`sector_sizes` 只在 `policy.is_lrc` 时必需；twobasis 支：`rank_twobasis_schedules` → `build_twobasis_memory_circuit` → `audit_twobasis_static`（lowered profile 下仍审计 uniform-legacy 发射，同 LRC 规则）；`MemoryExperimentPair.sec_policy_id` = `ranked.schedule.source` | 一个调用方漏传 `layer_blocks` 会静默走回退着色（深度仍 w，但层集不同）——测试 grep 全部调用点 |
| 调用方 | worker / pipeline / qualification / cli dem-smoke / export_stim 都传 `sec_policy=verification.sec_policy, layer_blocks=compiled.layer_blocks` | 同上 |
| 协议冻结 | `VerificationConfig.sec_policy`（to_dict/from_dict；`verification_from_dict` 缺键 → 默认 twobasis）；快照 `sec_policy_id` / `sec_candidates` / `sec_cert_iters`（`sec_synthesis` 镜像同值）；strict 要求 `DEFAULT_SEC_POLICY`；非 strict 后缀 `+sec{id}` / `+secK{K}I{iters}S{seed}`；resume 逐键比对 | v3.4 / SEC 前的 v3.5 快照被静默续跑 |
| validator | `FORMAL_SEC_POLICY="twobasis_nonint_v1"`、`FORMAL_SEC_CANDIDATES=16`、`FORMAL_SEC_CERT_ITERS=400`、`FORMAL_SEC_CERT_SEED=0`；核快照三键 + `verification.sec_policy` 四字段 + run config + 每条 candidate 的 `memory_experiment.schedule`（`sec_policy_id` / `candidates` / `cert_iters` / `depth`） | launcher 自称 v3.5 但候选在别的 SEC 下测 |
| CLI | `--sec-policy {twobasis_nonint_v1,lrc_ranked_v1}`（dem-smoke / slow-verify-seed / slow-verify-queue / stim-seed / stim-queue / run-campaign），默认 twobasis；`--strict-turns` + 非默认策略 → SystemExit | strict 下悄悄跑 LRC |
| launcher / ops | v35 三件不加旗标：`V35_DEFAULTS_OK` 追加 `DEFAULT_SEC_POLICY.sec_policy_id == 'twobasis_nonint_v1' == FORMAL_SEC_POLICY` 与 `(sec_candidates, sec_cert_iters) == (16, 400)`（本地编辑，**未 rsync**）；runbook 第 3 节同样断言 | 陈旧部署在 `_v35` 路径下冻结 LRC |

**没有改的**：解码器（BPOSD 50/10）、shards / watchdog / 重试（16 / 128、8 s/shot、3600/1800/1800、0/1）、plpr 公式、固定 T=12、prompt、种子、family 与 seed 语法、racing 关闭、结构保留通道、`DEFAULT_SEARCH_ROUNDS=12`；LRC 代码不删。

**可比性**：twobasis 产物与 v3.4（LRC + idle）及同日早些时候 SEC 仍为 LRC 的 v3.5 噪声版都**不可合池**；NSCC 的 v3.5 尚未在 SEC=LRC 变体下正式起跑（runbook 顺序是等 v3.4 canary 结束再上传），若任何 `_v35` 目录在本节落地前已首跑，`--resume` 会按协议不匹配拒绝，须换新目录（R22）。

**测试**：`tests/test_v35_twobasis_sec.py`（层：结构路径层数 = 最大度、每层 matching、覆盖恰一次，hgp 回退；电路：四 family × 两基 noiseless DEM、detector / observable / 测量 / tick 计数、每 check 每轮 w 个 CX、无 X–Z ancilla CX；噪声保真：paper profile 的 DEPOLARIZE1 只跟 reset、每 CX 后 DEPOLARIZE2、测前翻转、无其它噪声，legacy / si1000 / helios smoke；LRC 逐字节对拍改动前快照；认证：两序 d_ext 不同时选大者、delta-one 淘汰、确定性、metadata；端到端 BB72 默认策略 + BPOSD 200 shots 返回 ok；冻结：快照键、strict 拒绝 lrc、validator 常量、CLI 默认、`VerificationConfig` 往返）、`tests/test_hpc_v35_scripts.py`（扩：`V35_DEFAULTS_OK` 钉 SEC 策略、launcher 无 `--sec-policy` 旗标、runbook / 本文 / 02 稿引用）；`tests/test_sec_golden.py` 不变且绿。

### 2026-09-03 追加（八）：`telescoping_structural_v3_6` —— 论文 telescoping decoder（CPU 配置）、按预算点策略、误杀机制修复、coverage 仪表

来源：05 稿（大码公平性审计）与 06 稿（改进方案，用户 2026-09-03 拍板：大码按预算点放宽、未解出的 shot 计为失败、种子默认不加；同日晚定案 **decoder 换成 Mitten 论文 telescoping 的源码，其余不动或按其适配**）。权威顺序仍是代码 > 本文 > 06。**协议 id `bposd_structural_v3_5` → `telescoping_structural_v3_6`**（同日更早的自研级联草案 `cascade_structural_v3_6` / `relay_cascade_gari` 已删除、从未上机，其前缀与 `bposd_structural` 同为退役前缀），旧 run 不可合池、`--resume` 按名拒绝（`workflow/state.py::assert_resumable_protocol`）。

**不变**：噪声 `uniform_paper_v1` p=0.5% 单点；T=12；SEC 电路 `twobasis_nonint_v1`（两基不交错、detector 两基都发）；16/128 shards；racing 关闭；结构 (k, d_search) 保留通道与 3S+1R 慢位、8S:4R 父代、fixed-T 终局；plpr 纵轴；prompt 三条约束。

**Decoder**（02 稿 §11.4）：`DecoderName.TELESCOPING="telescoping_cpu"`（CASCADE 删除），`VerificationConfig.decoders` 默认 `(TELESCOPING,)`；`config.TelescopingDecoderConfig`（preset `telescoping_cpu_v1`，包 `telescoping-decoder@c97099b759dc`：S1/S2 关、`s3_system="gari"`、S4 关、`nc_policy="fail"`、S3-A/B/C 参数与 `variant_tables="package_default"`、`n_procs=1`；`to_telescope_config(base_seed=)`、`worst_case_iterations()`）、`TELESCOPING_PROTOCOL_FIELDS`（全部字段，与预算点无关）、`TELESCOPING_VENDORED_COMMIT`；`third_party/telescoping-decoder`（论文源码 vendored，MIT，`VENDORED_COMMIT`，editable 安装，C kernel 编译进 `$XDG_CACHE_HOME/telescoping_decoder`）；`verification/decoders.py::TelescopingObservableDecoder`（`telescoping_decoder.TelescopingDecoder` 的薄适配层：S3-A/B/C → NC **计失败**，`last_declared_failures`；seed = (shard seed, shot 序号)，跨机器逐 shot 一致；`close()` 释放包的子进程）、`build_observable_decoder(..., detector_kinds=, basis=, stage=, seed=, circuit_text=, telescoping_artifacts=, workdir=)`、`materialize_telescoping_artifacts`（build 时把两基 GARI 系统物化到 bundle 旁）、`decoder_dependency_report()` 版本含 `telescoping_decoder` / `telescoping_commit`；`verification/gari.py` 删除（`detector_kind_counts` 移到 `verification/circuit.py`）。发射器（`sec/stim_builder.py::_emit_circuit`、`twobasis_builder`）经 `detector_kinds_out` 记录每 DETECTOR 的 check 类型（电路文本逐字节不变），`MemoryExperiment.detector_kinds` + `to_metadata.detector_kind_counts`，DEM bundle 带 `detector_kinds` 与 `telescoping` 键（两基 GARI npz 相对路径 / 形状 / nnz / 指纹 / build 秒数；旧 bundle 为 None）。worker 结果每 (decoder, basis) 带 `decode_stage` 与 `metrics.stage_counts / stage_failures / stage_time_s / declared_failures / nc_policy / systems / system_source / package / gari / label_counts`（键 `S3A` / `S3B` / `S3C` / `NC`），`sampling.merge_shard_results` 按基求和。

**按预算点策略**（`config.budget_policy(n_budget)` → `{large_budget, search_shots, final_shots}`，快照键 `n_budget` / `budget_policy` / `decoder` / `telescoping`）：≤288 → search 20k / final 200k shots/基；≥300 → 5k / 50k；**没有逐级时限**（S3 各级按迭代数封顶）。decode watchdog = 下限 + 模型：max(1800 s, 8 s/shot × shots × max(1, 两基 DEM 列数 / 60000), shots × telescoping 最坏一枪)（`evaluation/pool.py::effective_decode_shard_timeout_s` / `telescoping_worst_case_shot_s`：`TELESCOPING_COST_S_PER_COLUMN_ITER` × `TELESCOPING_GARI_COLUMN_FACTOR` × 列数 × `worst_case_iterations()`；列数由 `run_build_dem_request` 结果的 `num_matrix_columns` 提供；`job_config.decode_budget_model` 记录 `telescoping_worst_case_shot_s` / `telescoping_term_s` / `cost_s_per_column_iter` / `gari_column_factor`）。strict 下 `--slow-shots/--final-shots` 偏离策略 → SystemExit；`run_campaign` 把 `spec.qubit_budget` 传给 `build_protocol_snapshot(n_budget=)`；终局 `evaluate_promotion_queue(..., stage="final")`；validator `hpc/validate_fig2_campaign.py` 钉 `decoders == ["telescoping_cpu"]`、每个 `TELESCOPING_PROTOCOL_FIELDS` 键、shots vs 策略（三键）、SEC 四个新字段、`resolved_versions.json` 含 `telescoping-decoder`（版本串）与 `telescoping_commit == TELESCOPING_VENDORED_COMMIT`（`workflow/checkpoint.py`；ldpc / relay_bp 不再要求）。

**误杀机制修复**（06 稿 §3）：B3 近重复门改 `code_hash` + pynauty 规范哈希（`memory/fingerprints.py::canonical_code_hash`，WL 哈希只记录不作门；结构行 / 排名行持久化 `canonical_hash` / `wl_hash`）；B4 reducer 每 (k, d_search) ≤2 实现（`reducer_id="structural_proxy_pareto_v2_coordcap"`，`coordinate_cap=2`），slot 3 跳过已选坐标，父代按唯一坐标轮转（`ranking.structural_coordinate_round_robin`）；B5 `decode_infeasible` 终局可重试一次（`CampaignState.decode_infeasible_hashes` + `<run_dir>/decode_infeasible_hashes.json`，终局行 `retry_at_final`，pool 对其打 `stage: "final"`）；M1 层序认证分级（迭代 = max(400, 4·核维)、3 seed 取 min、top-4 复认证 2000 iters、d_ext ≤ d_code_proxy 钳制、`converged_x/z`；`SecPolicyConfig.sec_cert_seeds/sec_cert_iters_scale/sec_refine_top/sec_refine_iters`，synth cache key 覆盖全部策略字段）；M2 `degraded_ext` 只作审计标志（`ranking.is_degraded` = `schedule_catastrophic` 唯一硬隔离，`is_flagged_degraded` 为审计视图）；M3 群阶上限 = `n_budget // 4`（`MAX_GROUP_ORDER=200` 改资源护栏，`catalog.set_group_order_budget`，contract 行 + failure action）；M4 worker 数按 RSS 封顶（`OMNIQEC_WORKER_RSS_BYTES` / `OMNIQEC_TOTAL_MEM_BYTES`，worker 结果带 `ru_maxrss`）；M5 每候选只先提交 noise-0 build、落地后再提 distance 与其余 build + `verification/circuit.py` synthesis 文件锁；次要：b≥288 注入 BB144 resident baseline、ledger scope 用 `n_total`、known_codes 加 Mitten 150/200、目录同构去重（`C{c}xD{m}` 的 metacyclic 重拼写被拒并给指针）。

**generator 信息**（06 稿 §4）：只上机器规则（B4 父代轮转）与 coverage 仪表（`construction/structure.py::structure_key` 加形状对与预算利用率分箱，`memory/coverage.py::catalog_gaps / coverage_fact_lines` 输出"本预算下从未提出的（分支 × 形状）"与每区 best_k / best_d_search，进策略简报与 Policy S-9 的 `fact_lines`，无 maximize/prefer 词，`test_prompt_neutrality` 不改）；种子与文献事实块默认不加。

**上机**：`hpc/nscc_claude_grid_v36.pbs` / `submit_claude_v36_grid.sh` / `claude_v36_status.sh`（`DECODERS="telescoping_cpu"`；无 `--slow-shots/--final-shots`、无 `FINAL_SHOTS_OVERRIDE`；`V36_DEFAULTS_OK` 自检断言 profile / decoder / `TELESCOPING_PROTOCOL_FIELDS` / 三键策略 / `import telescoping_decoder` 且包旁 `VENDORED_COMMIT` 含钉定 commit；序言 `module load gcc` + `export XDG_CACHE_HOME=runs/.cache` + `python -m telescoping_decoder.build` **预编译 C kernel 一次**，worker 继承、不编译，失败 exit 4；状态视图显示 `telescoping` preset）；`hpc/requirements_fig2_nscc.txt` / `requirements.txt` / `hpc/README_FIG2_NSCC.md`：`pip install -e ./third_party/telescoping-decoder`（需 gcc；rsync 必须带 `third_party/telescoping-decoder`），relay-bp 降为可选。

**测试**：`tests/test_telescoping_decoder.py`（替代同日删除的 `test_cascade_decoder.py`）、`test_v36_protocol.py`、`test_v36_retention.py`、`test_v36_sec_catalog_coverage.py`、`test_hpc_v36_scripts.py` 新增；`test_v35_*`、`test_campaign_*`、`test_v34_decode_budget.py`、`test_autonomy_mechanisms.py`、`test_measurement_completeness.py` 等按 v3.6 期望更新；`test_sec_golden.py` 不变且绿。

## 11. 已知残留

| # | 事项 | 性质 |
|---|---|---|
| R1 | Generator prompt 不进协议快照（§8）。已知一次实际变更：2026-08-26 `catalog_summary()` 加入 D_n 索引约定（§10 末节），跨该日的 run 比较须人工标注，快照不会替你挡 | 可比性口子 |
| R2 | 计划校验报错已可执行，但不保证模型遵守；若复发，下一个杠杆是给这一类校验单独提高修复预算 | 未验证 |
| R3 | `final_slow` 的 03 §9 扩展未实施：schedule 敏感性审计、高阶 BPOSD 对照、Tier-2/3 同构检查、slope_offset 重校准（`final_slow_summary.json` 的 `pending_extensions` 就是这四条） | 设计稿已定、未实现 |
| R4 | worker pool 单节点；多节点 PBS 分配会闲置 | 架构限制 |
| R5 | 序贯停止偏差不校正，仅在 Methods 披露 | 已知取舍 |
| R6 | 电光云上跑的仍是改动前的旧代码 | 待同步 |
| R17 | v3.5 当前只有 NSCC Claude 生产路线已接入：`nscc_claude_grid_v35.pbs` + `submit_claude_v35_grid.sh` + `claude_v35_status.sh`（v3.4 三件头部标 HISTORICAL，与 v3.3 三件、`nscc_claude_grid.pbs` 一起保留为历史，不得在当前代码下提交）。`fig2_campaign.pbs`、GPT/DeepSeek 臂、无调度器 Pod 平台与 DianGuang launcher 仍绑定 v3.2 或更旧快照；同步当前代码后不得不加改造就提交它们。要生成可与 Claude v3.5 合池的其他臂，必须逐一接入 search 16 / final 128 shards、按 shot 预算的 watchdog（8 s/shot）、`timeout_retries=0 / transient_retries=1`、`measurement_blocked` / `decode_infeasible` 语义、显式 `--noise-profile uniform_paper_v1 --p 0.005` 与 v3.5 validator | 平台覆盖缺口 |
| R18 | 三臂已有结果（beam/lsd + T=d + 旧筛选）、v3.2 结果与 v3.3 结果都**不可直接混比**。v3.2→v3.3 虽然 BPOSD/结构选择尺不变，但 shard-index RNG 流、超时重试与完整性门改变了可发布样本集。同一张 Fig.2 要么各臂全用 v3.3 重跑，要么显式分栏标注 | 可比性口子 |
| R19 | `explore_temperature`（R12）与结构通道无关：结构父代按 reducer 的 channel-neutral 顺序取，不经过 Boltzmann 抽样。接线 R12 只会影响 runtime 那 4 条 | 已知取舍 |
| R7 | Claude 订阅席位经 CLIProxyAPI 上 NSCC 的路线已成型，但 `-claude-login` 仍是登录节点上的一次性**人工**步骤；`--provider claude` 不可用，实际走 `--provider openai`。v3.3 提交器会拒绝同 `RUN_TAG`/预算点的已活动作业，launcher 也取 `flock`，但节点缺 `flock` 时仍只降级为警告。操作者必须在计算节点探针通过后才提交 | 需人工前置 |
| R20 | v3.4 仍没有跨 PBS 的 persistent shard ledger（`persistent_shard_ledger=false`）。watchdog 超时不再重放（`timeout_retries=0`）；瞬态错误（`subprocess_error` / `dem_load_error`）只在当前 `run-campaign` 进程内同-request 重放一次。只因超时而不完整的候选以 `decode_infeasible` 终态退出（不停止该点、不重排队）；其余不完整仍 `measurement_blocked` 并停止该点。人工重交不能被声称为“仅补缺 shard”，除非先实现并审计持久 shard 账本 | 已知恢复边界 |
| R21 | v3.4 产物（`_v34` 目录，`uniform_legacy_v1`：LRC 逐 tick idle 噪声——Mitten [[200,40,12]] 的单比特噪声暴露为论文模型的 9.6×、BB144 1.8×）与 v3.5 产物**不可合池**，也不可用 `--resume` 续到 v3.5（快照缺 `noise_profile` / `noise_values`）。v3.4 的 s/shot 实测（0.65–0.94 / 5.2–5.5）是 idle 噪声下的数字，只作 v3.5 的上界，v3.5 canary 须重测。去 idle 后 LRC 与论文单边 SEC 在 p=0.005 上的剩余差距是用户保留 LRC 的已知代价，不在 v3.5 范围 | 可比性口子 |
| R22 | v3.5 SEC = `twobasis_nonint_v1`（追加（七））的可比性与未调优面：(a) `d_circ_ext` / `degraded_ext` 与 `lrc_ranked_v1` 产物**不可比**（residual 集不同、100 vs 400 iters），evidence blob 以 `sec_policy_id` 区分；(b) 同日早些时候 SEC 仍为 LRC 的 v3.5 噪声版快照（`sec_policy_id=lrc_ranked_v1`、缺 `sec_candidates` / `sec_cert_iters`）被 resume 拒绝，须换目录；(c) si1000 / helios 在 twobasis 时间线上只保证正确性（X-memory 第 1 轮双 H 未合并、无 window hoisting、idle 记账未调优），不得作硬件阶段结果；(d) 认证成本 ≈ LRC 合成的 8×（BB144 ≈ 77 s，n=288 / w=9 每候选数分钟），synth cache key 变更使所有旧 `synth_*.pkl` 一次性失效；(e) `sector_h` 在两条策略下仍 `NotImplementedError`（R16），`layer_blocks` 不感知 mask；(f) hook 错误未被消除，只被精确认证并在基内层序间择优 | 可比性口子 / 未调优 |
| R8 | 坏侧 prompt 文本仍是手写散文，与校验器规则非同源；理想是从校验代码生成（registry 已对语法示例这么做） | 低优先级 |
| R9 | 预算压力与反思记忆的效果未经实测；需要新 campaign 与旧数据对比才能判定是否真的抬高了预算利用率 | 待验证 |
| R10 | 降本对照只有 2 轮 × 单样本，策略自身的方差（如提案数 64 vs 24）与改动效果混在一起；orchestrator 侧指标（往返数、输入 token、缓存命中）才是干净的 | 样本不足 |
| R11 | OpenRouter `:batch` 变体（五折）只能走异步 `/api/beta/batches`，同步 chat 调用返回 404，交互式工具环用不了 | 已排除 |
| R12 | `explore_temperature` 已是 `TurnPlan` 字段，过 [0,5] 校验、写进 `plan.json` 与 `progress.csv` 的 `explore_temp` 列，但**没有任何调用方把它传给 `archive.parent_records`**——唯一调用点 `memory/archive.py::feedback_payload` 用的是默认 `temperature=0.0`，`_boltzmann_order` 因此在生产路径上恒为恒等映射，parent 选择永远是确定序。该分支只被 `tests/test_autonomy_mechanisms.py` 覆盖。§10「自主性机制」那一行记的是机制存在，不是它已接线 | 已实现、未接线 |
| R13 | 03 §12 要求把 `funsearch.py` 与 `campaign.py` 一并拆进 `workflow/` + `orchestrator/` + `evaluation/`；实际只拆了 `slow.py` / `slow_worker.py`，前两者仍在包根且承重（Generator 与 fast 环；fast/slow 粘合层）。§1 的分层图记录的就是这个实际形态——设计条目被实践取代，不是代码欠账 | 设计稿已被实践取代 |
| R14 | 03 §4.2 列了一个 `generate(brief, n)` 的 policy 档工具，注册表里没有这个工具（现有 12 个见 §4）。生成由 `workflow/graph.py` 的 execute_plan 节点每轮驱动一次，不是 agent 可调用的动作 | 设计稿已定、未实现 |
| R15 | family 整数配额只在 `turn_audit.json` 里记录请求 / 产出 / 缺口（`family_quota_*`），**不强制**——generator 以自由程序作答，某个 family 合理落空不该让整轮失败；代价是持续偏斜只能靠人翻审计文件发现，而该文件只在 `--strict-turns` 下才写 | 已知取舍 |
| R16 | 02 §6.4 的 (check_type, qubit_sector) 门型表未实施：`verification/circuit.py:156-159` 对任何非平凡 H-mask 直接 `raise NotImplementedError`，`sec/stim_builder.py` 无 `h_mask` 路径。构造与 fast/structural 阶段照常携带 mask（进 `code_hash`、导出 `h_mask_weight`/`h_mask_sha256`），只有电路阶段拒绝——失败形态是**逐候选静默降级**：`evaluation/pipeline.py:199-206` 记 `circuit_error` 后 `continue`、`evaluation/worker.py:78-79` 记 `build_error`、`:213-216` 记 `circuit_error`、`cli.py:770-772` 记 `status:"error"`。因此 `--deformation-mode sector_h` 不会显式失败，而是烧完预算产出一整轮无 DEM、无 LER 的 `circuit_error`，退出码仍为 0；防呆应在参数处理期前置硬失败（详见 02 §14 第 1 条、01 §13） | 设计稿已定、未实现 |

---

## 附：模块速查

| 包 | 职责 | 关键文件 |
|---|---|---|
| `workflow/` | LangGraph 图、状态、协议快照、检查点 | `graph.py` `state.py` `checkpoint.py` |
| `orchestrator/` | policy agent、计划 schema、工具注册表、prompt、手册 | `loop.py` `plan.py` `registry.py` `manual.py` `prompts/` |
| `tools/` | read/draft/policy 三档工具，docstring 即手册 | `queries.py` `ledger_ops.py` `precheck_draft.py` `acquisition_score.py` |
| `memory/` | archive、ledger、回测器、bandit、指纹、已知码、覆盖度 | `archive.py` `ledger.py` `backtester.py` `coverage.py` |
| （包根） | 成本聚合、两个 CSV 导出 | `token_usage.py` `progress_csv.py` `summary_csv.py` |
| `construction/` | 四个 family 插件、群目录、种子语法、结构坐标 | `registry.py` `lp.py` `bb.py` `hgp.py` `structure.py` |
| `sec/` | twobasis 层构造 / 基内序认证 / 电路，LRC 综合（历史），边着色，共享 stim 发射器，审计 | `layers.py` `twobasis_rank.py` `twobasis_builder.py` `stim_builder.py` `schedule.py` `schedule_rank.py` `coloring.py` `audits.py` |
| `verification/` | 电路、解码器、DEM、距离、噪声 profile、代理距离 | `decoders.py` `circuit.py` `distance.py` `noise.py` |
| `evaluation/` | 单候选流水线、作业池、DEM 缓存、采样/racing、证据、回写、反馈 | `pipeline.py` `pool.py` `sampling.py` `evidence.py` |
| `llm/` | completion-only 客户端 + 超时/重试 + usage 归一化 | `retry.py` `deepseek.py` `openrouter.py` `factory.py` `usage.py` |
| `hpc/` | 按论文设置的预设启动入口（PBS）、通用工具、状态视图、Cloudflare 网关（开源分支不含历史部署脚本与运行手册） | `nscc_claude_grid_v36.pbs` `submit_claude_v36_grid.sh` `submit_gpt_v36_serial.sh` `nscc_baseline_grid.pbs` `validate_fig2_campaign.py` `worker_deepseek/` |

# OmniQEC Fig.2 重构设计定稿 03（v2.1）：Workflow 重建——Lab-in-the-Loop、Orchestrator 与学习层

- 状态：**定稿 v2.1**（2026-08-15）
- 沿革：v1.0（方案 B 草案）→ v2.0（用户扩展稿：lab-in-the-loop + Orchestrator agent + 学习层）→ **v2.1 定稿**：并入两项方法学修正（§6.1 前瞻回测、网格冷启动规则）、统一单框架修正（双模式删除、混合 family 搜索）、LangGraph 工程核查修正（§2.3）、文献定性修正（§0 关键文献）、两层指纹方案（§6.3）。
- 依据：PDF §4–§6 + 设计稿 01（定稿，已实现）+ 设计稿 02（定稿 v1.1）+ OmniQEC v1 论文 + 五轮文献调研与讨论决策
- 关键文献（已核实，含修正后定性）：
  - **Robin**（FutureHouse），Nature (2026), DOI 10.1038/s41586-026-10652-y——lab-in-the-loop 骨架先例（附注：其头条效应量经人工复核 7.5×→1.75×，支持本设计"curator=确定性回测器"的立场）
  - **ACE**，arXiv:2510.04618（ICLR 2026）——evolving playbook、Generator/Reflector/Curator、增量 delta 更新与 grow-and-refine、brevity bias / context collapse；承 Dynamic Cheatsheet 的 adaptive memory 并扩展
  - **Gupta, Hartford & Liu**，Findings of EMNLP 2025（arXiv:2509.21403，单篇）——置换反馈消融证明裸 in-context 反馈在贝叶斯优化/实验设计任务上"不学习"；有原则统计方法胜过 naive LLM agent（范围限定于该类任务）
  - **ShinkaEvolve**，arXiv:2509.19349（Sakana AI）——fitness/novelty 加权 parent 采样、新颖性拒绝采样；UCB1 bandit 原文用于 **LLM ensemble 选择**，本设计移植到策略臂
  - **GEPA**，arXiv:2507.19457（ICLR 2026 Oral）——反思式 prompt 进化（本设计二期）
  - **Co-Scientist**（Google），Nature (2026), DOI 10.1038/s41586-026-10644-y——Elo 锦标赛 + 成对多轮辩论排名，**明确弃用**（本系统双环皆有真值裁判）
  - **Cruz-Benito, Cross, Kremer, Faro（IBM）**，arXiv:2606.02418——LLM 引导的 BB 码进化发现（**code-capacity only**）；其彩色 Tanner 图规范形（BLISS via python-igraph）+ local-Clifford 二层核验与"distinct 计数为保守上界"的报告口径，为本设计 §6.3 的先例（非社区认证标准）
  - **Liu–Marquardt SCE**（Structured Concept Evolution），arXiv:2606.24808——LP 码概念进化（含非阿贝尔群；**code-capacity only**；其 WL 图核指纹为 §6.3 Tier-1 先例）。两篇竞品皆无电路级证据——本框架的电路级 LER 是**对领域现状**的差异化
  - **LangGraph 1.x**（1.0 GA 2025-10-22，semver 承诺至 2.0）
- 范围：**一稿全定**——workflow 层（LangGraph）、Orchestrator 与工具体系、技术手册、学习层、evaluation/ 库、adaptive sampling、decoder qualification、fast loop/archive 适配、legacy 清理。
- 替代关系：全量替代 `funsearch.py` 循环编排与 `campaign.py`；`slow.py` 拆分为 `evaluation/`；`postsearch.py`/Sinter 删除（迁移门后）；legacy 删除；README 全量重写。

---

## 0. 决策记录（v2.1）

| # | 决策 | 来源/锚点 |
|---|---|---|
| D1 | **单一完整框架**：Orchestrator（LLM policy agent）为常驻中枢，fast/slow 两环 top-k 全部回流给它；无"双模式"，无 scripted 科学消融臂（scripted 仅为工程降级路径：dry-run/CI/无 key 跑通管道） | 用户定案（Fig.1 即系统） |
| D2 | 骨架 = **lab-in-the-loop**：query → Orchestrator → fast loop → slow loop → 证据回流 | Robin 同构 |
| D3 | **混合 family 搜索**：campaign 单元 = **预算点**，四 grammar 默认全开，LLM 在混合空间搜索并跨 family 学习；per-family 结论 = 事后归因；spec 的 family 字段可取子集（受控对照仍可表达，非默认） | 用户定案（改写 v1.0"每 family 独立 campaign"） |
| D4 | 两层分权：policy（Orchestrator）/ protocol（冻结确定性模块）；分权表 + 工具层断言 | 讨论定案 |
| D5 | 框架 = **LangGraph 1.x 锁定** + 自研 `llm/` 客户端；仅 `workflow/` import langgraph（langchain-core 作为其硬依赖随装，但不用 langchain 包与模型集成） | 核查修正 |
| D6 | 技术手册 = ACE 式 evolving playbook（静态/动态/技能三区）；**curator = 确定性回测器（非 LLM），且 supported 判定仅用提案后数据（前瞻验证）+ 并发假设上限** | ACE + 方法学修正 |
| D7 | 学习层一期 = insight ledger + 假设标签 + novelty/已知码 + 策略 bandit；**校验用内生指标（insight precision 留出代）**，campaign 级消融（置换反馈、agentic vs scripted）移出 v1 范围 → future work | 用户定案 |
| D8 | racing 固定 **anchor p（按 profile 取：uniform 0.005 / si1000 1e-3 / helios scale=1）**；全 p 扫描与 slope survivor-only；**slope_offset 在 LRC 下重校准**（0.958 为 ver1 单边模型校准量，随 02 §2.0 失效） | 论文协议 + 修正 |
| D9 | LLM 锦标赛/辩论排名弃用；晋级 = acquisition 基准 + agent 决策 + 护栏 | Co-Scientist 反例 |
| D10 | decoder 默认集 =（beam, lsd）承 02 D6；qualification harness 独立 CLI 不进图 | 02 定稿 |
| D11 | postsearch/Sinter 删除；**删除前 Sinter 独立裁判对拍（迁移门）** | 评审定案 |
| D12 | hook surrogate 退役；tie-break 用 (Δmin, τA) fast/slow 双变体，列名区分 provenance | 02 §10 + 评审 |
| D13 | **Fig.2 网格（各预算点）关闭跨 campaign 全局库**：预算点间严格冷启动、独立可比；全局库为 assistant 使用与后续研究的特性 | 方法学修正 |

继承不变：evaluation/ 零框架依赖；E5/E6 口径、Wilson CI、shard 的测量协议地位；`workers=0 → cpu-1`、BLAS 单线程。

## 1. 总体架构：两层分权

```text
policy 层    Orchestrator agent —— 决定"测谁、花多少、往哪探、何时停"
               │  只能经注册表工具行动；全 transcript 落盘
protocol 层  冻结的确定性模块 —— 决定"怎么测、测出多少"
               sec/(02) · construction/(01) · evaluation/(§7) · verification/
```

**分权表（定稿后不经修订不得扩权）**：

| Orchestrator 拥有（policy） | 冻结（protocol，agent 无写权限） |
|---|---|
| 每轮程序数与策略配比（含 **family 注意力分配**，上限内） | SEC 合成/排名/审计（02 全部） |
| 晋级名单与预算档（pilot/confirm；名单有界） | 噪声 profile、decoder 配置 |
| 账本 propose/retract（回测器终裁） | racing/shots/CI 口径、anchor p |
| 主动 finalize（≥ min_turns 后） | 证据通道、archive/HoF 更新规则 |
| 向用户澄清/汇报的时机 | 指纹/查重、隔离与 degraded 语义 |
| | 终局报告的一切数字来源 |

**三条铁律**：

1. `evaluation/ sec/ construction/ verification/ tools/ orchestrator/` 零框架依赖（pydantic 允许）；仅 `workflow/` import langgraph——换框架 = 重写一个薄层；
2. 图 state 只承载轻量指针与指标；矩阵/DEM/shots 等重产物留文件系统；
3. **agentic 的是决策，可复放的是测量**：agent 层以下的种子纪律、预注册 shard 流、确定性 decoder 全部保持；agent 的每个决策连同 rationale 进 transcript。LLM 决策的随机性是方法的一部分，以 transcript + 协议冻结 + 全链路种子化保证可审计，论文方法节声明。

## 2. Workflow 层（LangGraph）

### 2.1 图结构

```text
[intake]──spec 确认（交互式 interrupt() 或 CLI 直给）──▶ 协议快照冻结
    │
    ▼
┌──────────────────── campaign loop ─────────────────────┐
│ [orchestrator_turn]  手写工具环（§3.2），产出 TurnPlan   │
│        │                                                │
│ [execute_plan]  generate → sandbox → fast_eval           │
│        │        → novelty 闸门 → promote(plan)            │
│        │        → slow_verify（零 agent 区，§8）           │
│        ▼                                                │
│ [reconcile]  archive/HoF · 账本回测 · bandit · digest     │
│              （fast top-k + slow top-k 双路回流 ▶ 下一 turn）│
└── 循环；checkpoint 每 superstep；finalize 退出 ───────────┘
    │
    ▼
[final_slow]（§9 全协议）──▶ [report] ──▶ END
```

slow_verify 内部零 agency——那里是纯测量，agent 通过工具**消费**慢环证据而不介入它。reconcile 的 **campaign digest 显式含两路 top-k**：fast top-k（Q_eff/结构特征/失败签名）+ slow top-k（LER/CI/距离证据/审计标志），是 Orchestrator 下一 turn 的标准输入（对应 Fig.1 的 Memory/Feedback 职能）。

### 2.2 State schema（workflow/state.py）

```python
class CampaignState(TypedDict):
    run_id: str; turn: int
    run_dir: str
    spec: dict                    # CampaignSpec（冻结；families 默认全四族）
    protocol: dict                # 协议快照：sec_policy_id / decoder / noise / anchor_p(按 profile) /
                                  # racing 参数 / P / T_Δ / bandit / 工具预算 / max_turns
    archive_path: str; queue_path: str
    manual_dir: str; ledger_path: str; fingerprint_index_path: str
    hof_summary: list[dict]       # 轻量：code_hash / family / plpr@anchor / CI
    baseline_summary: dict        # resident baseline @anchor + 全曲线指针
    bandit_stats: dict            # 各策略臂 UCB 统计（策略 × family 维度）
    budget: dict                  # tokens / slow-shots / 墙钟 / turns 四本账余额
    plan: dict | None             # 本轮 TurnPlan（含 rationale）
    transcript_ptr: str
    rng_cursor: dict              # proxy ISD / 色置换计数器 / shard 流游标全覆盖
    stop: dict
```

### 2.3 Checkpoint / resume 语义（含工程核查修正）

- SqliteSaver（langgraph-checkpoint-sqlite），库文件在 run 目录，thread_id = run_id；离线 HPC 友好；**安全项：`LANGGRAPH_STRICT_MSGPACK=true`**（CVE-2025-64439 后官方指引）；
- **recursion_limit 显式设置 = max_turns × 4 + 64**（版本无关的确定性习惯；注：1.0.6 起默认已升至 10000，"默认 25 必死"的说法过时）；
- **durability="sync" 显式设置**（默认 async 崩溃有小窗口丢失；相对慢环成本可忽略）；
- checkpoint ≠ 完整 durable execution：resume 重执行被中断节点（官方语义"nodes after the checkpoint re-execute"；同 superstep 已完成兄弟任务经 pending-writes 不重跑）。药方三件：
  1. slow 侧幂等键 **(code_hash, profile, p, basis, shard_id)**——预注册流 + 确定性 decoder 使重算逐位相同，writeback 台账按键去重，execute_plan 入口先查台账 → slow 段节点内部可续；
  2. LLM 原始响应**先落盘再解析**（orchestrator_turn 与 generate 均是）；
  3. `--resume` 校验 state.protocol 与当前 config 一致，不一致拒绝续跑。
- 依赖锁定：langgraph 1.x 精确版本（**不用 langgraph-prebuilt**——发生过未约束破坏发布，且工具环手写）；跨版本 checkpoint 无官方保证（langgraph-checkpoint 在 1.x 期间历经 2→3→4 major）——**campaign 进行中禁止升级依赖**；resolved versions 快照进 run 产物。

### 2.4 Fleet 与 CLI

- fleet 单元 = **预算点**（D3）：Fig.2 网格 = 每个预算点一个 campaign，四 grammar 混合搜索；fleet 启动器顺序/并行拉起，**网格 run 间零共享**（D13 冷启动；例外：resident baseline 结果缓存可共享——键含 (code_hash, sec_policy_id, decoder+config, profile, p, basis, shard 区间)，确定性使共享逐位无损）；
- **LLM 全局限流**：客户端层 token bucket，多 campaign 共享配额；
- CLI：`run-campaign [--resume]`（spec 从命令行/文件给出即跳过交互 intake；同一图、同一 agentic policy）；`run-agentic-cycle`/`run-funsearch` 保名转发（后者 = 永不晋级档）；交互式 intake 的 `interrupt()` 需交互终端，PBS 批处理请用 spec 直给。

## 3. Orchestrator Agent

### 3.1 intake

- 输入自由文本或 CLI spec；输出 pydantic 严格校验的 **CampaignSpec**，字段白名单 = 论文 Fig.1 五项：qubit budget（必填）/ weight limit（默认 9，只可收紧）/ connectivity（v1 仅 all-to-all，其余显式拒绝并说明）/ code family（**默认四族全开**，可取子集）/ objective（v1 仅 plpr@anchor；其余记次要指标并明示不作为搜索目标）；
- 协议参数一律取默认；schema 外请求显式拒绝而非静默忽略；交互式时 `interrupt()` 回显确认 → spec 与协议快照冻结。

### 3.2 turn 结构（每轮）

1. 读手册（静态目录 + 动态区）与 state 摘要（含上轮 digest 的双路 top-k）；
2. 账本维护：`ledger_propose / ledger_retract`（回测器即时裁决）；
3. 产出 **TurnPlan** = {程序数, 策略配比, **family 注意力分配**, 晋级预算档, 定向指令（如"探索 C3⋊C4 附近"）, rationale}——工具层校验界内；
4. execute_plan：`generate(strategy_brief)` → 生成 LLM 产出 construct 程序（**LLM 永不直接输出矩阵**；程序 × 种子展开沿论文协议，默认 24 程序 × 20 种子；prompt 反馈上下文保留 feedback_mode 开关 both/fast/slow/none）→ sandbox → fast_eval（结构 / k / d_proxy / Q / Δmin_canonical / 指纹 / 已知码查重）；
5. agent 审清单：`acquisition_score` 取确定性基准，`promote(ids, budget_class)`——偏离 acquisition 序或 bandit 建议配比必须带 rationale（入账）；
6. slow_verify 按冻结协议执行（§8）；reconcile 确定性回写（archive / HoF / 账本回测 / bandit / digest）；
7. 终止：目标达成（beats-BB 且 CI 分离）/ 四本账任一耗尽 / agent 主动 `finalize`（≥ min_turns 后）。

工具预算：每 turn ≤ tool_budget 次调用（默认 32）；超限强制出 plan 或 finalize。

### 3.3 护栏

- 四本账硬上限：tokens / slow shots / 墙钟 / max turns；
- **工具层断言（非 prompt 恳求）**：晋级名单 ≤ promote_cap；audits 不可跳过；隔离（schedule_catastrophic / degraded_*）不可解除；协议字段物理上无写接口；
- 全 transcript 落盘；report 节点从 evaluation/ 产物渲染数字，**不经 agent 转述**。

### 3.4 工程降级路径（非科学模式）

`Policy` Protocol 接口保留一个 `FallbackPolicy`（Q_eff top-k + every-N 晋级，完全确定）——用途仅限 dry-run、CI、无 API key 时跑通管道与回归测试；不出现在论文叙事，不作消融对照（D7）。

## 4. 工具体系（tools/ + orchestrator/registry.py）

### 4.1 注册表装饰器

`@tool(scope="read"|"draft"|"policy")` 一次完成：① 签名 + pydantic schema → function-calling 定义（自研 `llm/` 客户端消费）；② 作用域强制；③ **docstring → 手册静态区自动渲染**；④ transcript 自动记录。

### 4.2 v1 工具目录

| 工具 | scope | 说明 |
|---|---|---|
| manual_read(section) | read | 手册分区读取 |
| query_archive(filter) | read | 结构化查询（如 "lp_nab 且 Δmin≥3 且未晋级"） |
| query_evidence(code_hash) | read | 单候选完整证据链 |
| precheck_draft(program) | draft | 草稿预检：k / d_proxy / 指纹 / 预算合法性 |
| fingerprint_lookup(x) | read | archive + 全局已知码的新颖性距离 |
| known_code_lookup(x) | read | 已知码表等价类查重（§6.3） |
| acquisition_score(ids) | read | 晋级基准分（确定性公式，版本化） |
| ledger_propose / ledger_retract | policy | 账本操作，回测器即时裁决 |
| bandit_status() / budget_status() | read | 策略臂建议与统计 / 四本账余额 |
| submit_plan / promote / finalize | policy | 三个受限写 |
| generate(brief, n) | policy | 触发生成 LLM（工具化的第二 LLM 角色） |

### 4.3 纪律

工具对 protocol 层零副作用；限时限额；返回一律是机器计算的真值——agent 的话永远不直接成为记录。

## 5. 技术手册（manual/）

- **static/**：模块与工具目录，注册表自动渲染，唯一事实源 = docstring；
- **dynamic/**：supported 洞见、bandit 权重快照、campaign digest、（assistant 用）全局库视图；
- **skills/**：产出过前沿码的 construct 程序片段 + 证据指针（Voyager 式；技能 = 可执行程序，验证 = 慢环证据）。

ACE 机制采纳：增量 delta 更新、永不整体重写（防 context collapse）；grow-and-refine，退役进墓地不物理删除；拒绝摘要化（防 brevity bias）。与 ACE 的关键差异（本设计强化）：**Curator = 确定性回测器**——条目入 dynamic/ 的唯一门槛是回测通过，LLM 只有提案权。

## 6. 学习层

设计原则（Gupta 教训，范围限定于其实验设计任务但教训通用）：裸 in-context 反馈不构成学习——**学习必须外化为经过验证的工件**。本层即"LLM 在迭代中学到规律并进步"的实现机制。

### 6.1 Insight ledger（含方法学修正）

```python
class Insight(BaseModel):
    id: str
    status: Literal["hypothesis", "supported", "refuted", "retired"]
    scope: Scope           # families ⊆ {bb,hgp,lp_ab,lp_nab} × budget 段 × profile
    predicate: Predicate   # 受限表达式，仅引用已计算字段
    claim: Claim           # {metric, direction, min_effect?}
    evidence: Evidence     # support_n / contra_n / effect_size / CI / last_gen
    provenance: dict       # proposer, turn, propose_gen, rationale, 被引计数
```

回测口径（协议参数）：

- **前瞻验证（反 double-dipping）**：supported/refuted 判定**仅使用 propose_gen 之后到达的慢环证据**；提案前数据只能维持 hypothesis。作用域内前瞻样本 N ≥ 20 方可判定；效应量 bootstrap 95% CI，方向一致且不含零 → supported；反向 → refuted；
- **多重性控制**：并发 hypothesis 上限（默认 16）；超限须先 retract 或等待判定；
- 回测在 reconcile 增量重算；refuted 即刻移出 prompt、入墓地（防复提指纹）。

### 6.2 假设标签与信用分配

construct 程序携带 `exploits: [I-*] / tests: [H-*]`；reconcile 记回实测结果。产物侧输出**"系统习得规则表"**（supported 条目 + 前瞻证据）——论文级工件。**学习有效性以内生指标校验（D7）**：insight precision = supported 规则在留出代上的预测有效率（零额外 run 成本）；campaign 级对照消融移出 v1。

### 6.3 Novelty 与已知码（两层指纹，先例对齐）

- **Tier-1（快筛，每候选必做）**：便宜不变量——n/k、行列度分布、群不变因子（01 稿 metadata 已有）、**彩色 Tanner 图 WL hash**（三色约定 qubit/X-check/Z-check；WL 指纹先例 = SCE 论文的 novelty penalty）→ 新颖性距离；
- **Tier-2（Tier-1 碰撞时）**：精确彩色 Tanner 图规范形（pynauty 或 bliss，可选依赖，缺失降级为 Tier-1 + 明示；先例 = Cruz-Benito 的 BLISS via igraph）；
- **Tier-3（finalist 级，可选）**：local-Clifford 等价核验；
- novelty 计数按**"置换等价下的上界"**口径报告（沿 Cruz-Benito 的保守框架）；
- **晋级闸门**：与已测候选指纹距离 < ε 的近重复不进慢环（precheck_draft 使近重复在提交前就不产生）；
- **已知码表**：BB 文献表、Mitten 表（依赖 01 稿 GAP `Elements(G)` 索引→元组转换工具，两处合并实现）、Zoo 预算内条目；重发现 → 定向反馈；
- parent 采样 fitness/novelty 加权（ShinkaEvolve 式，权重进协议）。

### 6.4 策略 bandit

臂 = {exploit, test_hypothesis, novelty, transfer, wildcat} ×（可选）family 维度；奖励 = 该臂候选的代际贡献（新 HoF/前沿改进份额）；UCB1 更新（机制借鉴 ShinkaEvolve，其原文用于 LLM ensemble 选择）；建议配比进手册动态区作知情信号——agent 可偏离（带 rationale 入账），每臂保底 ε_arm 防灭绝。混合 family 搜索下，**per-family elite quota（archive 现有机制）+ ε_arm 共同防单一 family 垄断**。

### 6.5 跨 campaign 全局库（只读；D13 约束）

冠军 genotype + 指纹索引 + supported 洞见（跨 scope 引用自动降级为 hypothesis 重新回测）。**Fig.2 网格 run 强制关闭**（预算点间独立冷启动，可比性优先）；开启场景 = assistant 使用与后续研究（暖启动：小预算冠军的 budget-lifting 变换进初始 archive）。

### 6.6 二期（明确不进 v1）

GEPA 式策略简报进化；meta-agent 工具合成（会击穿协议冻结）——future work。campaign 级学习消融（置换反馈、agentic vs fallback 同预算）亦归此类（D7）。

## 7. Evaluator 库（evaluation/）

```text
evaluation/
├── pipeline.py   # 单候选管线：sec 合成(缓存)→审计→DEM→解码→证据
├── pool.py       # work-stealing 进程池（语义不变）
├── dem_cache.py  # DEM 每 (candidate, p) 一次 + worker 端 decoder 对象缓存
├── sampling.py   # anchor-p racing 控制器（§8）
├── decoders/     # beam(默认)/lsd(保底)/bposd/relaybp(选项) 会话封装
├── evidence.py   # d_circ_ext/DEM-ISD/slope 合并、degraded_*、Wilson CI
├── ler.py        # E5/E6（原样保留）
├── writeback.py  # slow_results、rankings、archive reconcile、幂等台账
└── worker.py     # 子进程入口
```

修订三处：① sampling 实现 anchor-p racing，全 p 扫描与 slope 移入 final 管线；② slope 通道 survivor-only，**slope_offset 在 LRC 电路上重校准**（校准集 = final_slow 存活者 + baseline）；③ 其余不变：decoder 会话/DEM 缓存、BLAS 单线程、rounds = 完整 X 测量周期（02 staggered 约定）。

## 8. Adaptive sampling：anchor-p 双门槛 racing

- **anchor p 按 profile 取**（uniform 0.005 / si1000 1e-3 / helios scale=1，源自 02 §6.1 profile 默认；伪阈值自动选点进待办）；racing 仅在 anchor 点；
- Stage 0 pilot（默认 5k/basis）→ 存活判据（或语义）：CI 下界 < HoF@anchor 最优 UB **或** < baseline@anchor UB（竞选早期 HoF 空 → 事实上单门槛，淘汰率前低后高是预期行为）；每轮 +batch（默认 5k）；两门槛均被 LB 穿越 → 出局 `stopped_by=racing`；封顶 shots_max 或 max_errors（per candidate × basis × p）；存活者补至 shots_confirm；
- **门槛快照规则**：取 execute_plan 入口的 state 快照，批内静态——出局判定与 shard 完成顺序无关；
- 双 basis lockstep 追加；plpr 合成口径不变；CI 在原始失败计数上计算后传播；
- shard 流预注册 (seed, shards_max)，adaptive 只决定消费深度；序贯偏差不校正、方法节披露；stopped_by / 消费 shard 数 / 每阶段 CI 进 slow_results；
- baseline：不 racing，直接 shots_confirm × 全 p 曲线（fleet 共享缓存，§2.4）；
- sensitivity audit（02 §4.4）复用本控制器，预算减半档；
- 参数全进协议快照：anchor_p / shots_pilot / batch / shots_max / shots_confirm / max_errors / ci_z / ε / ε_arm。

## 9. final_slow 完整职责

对 top-K（含 beats-BB 存活者；per-family 归因所需的各族最优亦入列）：

1. **T = d 全轮数**重测（搜索期 Tsearch = max(3, min(d_hint, 8)) 仅为吞吐）；
2. 全 p 扫描 + slope 拟合（重校准后口径）；
3. BP+LSD 验证 + 高阶 BPOSD reference；
4. schedule-sensitivity audit（02 §4.4）；
5. 同构核验（§6.3 Tier-2/3 finalist 级）。

产出 final Fig.2 frontier + ranked package（Fig.1 输出：discovered code + compiled circuit + decoding performance；export-stim 薄导出承 §12）。

## 10. Decoder qualification harness（独立 CLI，不进图）

`qualify-decoders --set <路径> --reference lsd bposd_ref --candidates beam relaybp ...`

- set = 4 families × 3 sizes × 2–3 instances ≈ 24–36 码（固定种子 construct + LRC 电路，uniform + si1000 双 profile）；BB72/144 入 set 作锚点；set 定义进 docs/qualification/，版本化；
- reference = LSD(order-8 修复配置) + 高阶 BPOSD（CS-7 reference-only）跑满 shots_confirm；
- 统计口径：ρ 在 log-plpr 上计算；**主判据 = reference CI 分离码对的成对一致率 ≥ 0.9，且每 profile ≥ 2 个 p 档双档成立**；per-family ρ 仅诊断；family 偏差判据保留；
- 记录：LRC 电路实测吞吐、LER、一致率、per-family 偏差；**beam 确定性核验**与 num_results=32 吞吐在此定；
- 输出 qualification_report.json + markdown 摘要。

## 11. Fast loop / archive 适配

1. 证据接线：`degraded_ext` 与 `degraded_isd` 同权隔离；`schedule_catastrophic` 不测 LER、不进父代、进反馈；`schedule_sensitive` 只进产物与注解；`verified_circuit_distance = min(d_circ_ext, DEM-ISD, exact-stim)`，d_eff/Q_eff/d_cap 不变；
2. surrogate 退役（D12）：tie-break 双变体——fast 行 `delta_min_canonical`+`tau_a_canonical`（canonical 着色 + GF2 Δ=1 floor，毫秒级，不 gating），slow 行 `delta_min_top1` 等（P=64 全排名）；列名区分 provenance；
3. BB 盲测边界：prompt 永不提 BB；resident baseline 仅测量侧按预算注入；organic 重发现按普通候选 + §6.3 同构处理；`slow-verify-queue` 无条件注入改显式开关；
4. archive/HoF 结构不动；per-family elite quota 保留（混合搜索的多样性护栏）；HoF/baseline 摘要进 state。

## 12. 删除与清理清单

| 对象 | 处置 |
|---|---|
| postsearch.py / sinter_worker.py / verification/sinter_sweep.py / CLI sinter-* | 删除（D11）。**迁移门**：sec/ 生成同批 LRC 电路，新 evaluation/ 与 Sinter 同 LSD-0 对拍 plpr（CI 内一致）后执行 |
| CLI export-stim-* | 保留，重实现为 sec/stim_builder 薄导出（ranked package 与 qualification set 发布依赖） |
| runner.py / orchestrator.py / agents/ / proposal_eval.py / proposal_round.py | 删除（legacy；orchestrator.py 重名必清） |
| funsearch.py / campaign.py / slow.py / slow_worker.py | 拆解进 workflow/ + orchestrator/ + evaluation/ |
| surrogate/ | 退役（D12） |
| README 双语 | 全量重写（01–03 为规范源；Fig.1 家族列表与 decoder 列表同步更新入论文改稿清单） |
| PROGRESS.md / E6 状态文档 / hpc/ 旧脚本 | 归档说明 |

仓库最终形态：

```text
workflow/       graph.py / state.py / checkpoint.py       ← 唯一 import langgraph 的层
orchestrator/   loop.py / registry.py / policies/ / prompts/  ← 零框架
tools/          query_*.py / precheck_draft.py / acquisition_score.py / ledger_ops.py
manual/         static/(自动渲染) dynamic/(回测把关) skills/
evaluation/  sec/  construction/  verification/  llm/       ← 零框架
```

## 13. 实施顺序（跨 01–03 统一排期；01 已完成）

**Milestone A（Fig.2 关键路径）**：

1. sec/（02 §12 顺序，黄金样本先行）；
2. evaluation/ 机械拆分（语义不变）+ 新旧对拍 + Sinter 迁移门 → 删除清单前半；
3. anchor-p racing + baseline 共享缓存；
4. beam 接入 + qualification harness（可并行；decoder 预实验数据直接对拍）；
5. workflow/ 图 + state + checkpoint + FallbackPolicy 跑通管道（dry-run 级）；
6. final_slow 全协议 + slope 重校准。

**Milestone B（agentic + 学习层，不阻塞 A）**：

7. tools/ 注册表 + 手册静态区渲染 + 只读工具族；
8. 指纹/已知码表 → ledger + 前瞻回测器 → bandit → 手册动态区；
9. **agentic Orchestrator**（工具环）+ intake + ranked package 报告；

**收尾**：

10. 删除清单收尾 + README 重写；
11. 端到端回归（§14）。

硬规则：**B 的任何延期不得阻塞 A**；A 完成即可用 FallbackPolicy 开跑协议级实验，B 上线后正式 campaign 切 agentic。

## 14. 验证与回归

1. 机械拆分对拍（同 seed 同 shards 逐字段）；Sinter 迁移门；
2. 图层：kill -9 → resume 同 turn 恢复、台账续跑无重复写；protocol 快照不一致拒续；durability=sync 生效；dry-run 全图走通；
3. racing：人工悬殊队列断言出局轮次与 stopped_by；门槛关闭退化为固定 shots；快照规则顺序无关性测试；
4. qualification 报告样张（一致率 / 双 p 档 / 吞吐 / beam 确定性）；
5. 工具层：作用域断言、预算耗尽强制 finalize、transcript 完整性、docstring→手册渲染一致性；
6. 学习层（内生校验，D7）：前瞻回测正确性（构造已知规律的合成数据，断言 supported/refuted 判定）；insight precision 留出代计算；novelty 闸门与已知码查重（Mitten/BB 锚点码必须命中）；
7. 端到端：单预算点小规模 campaign（混合四 family）+ spec 直给与交互 intake 双入口，analyze-runs 对新产物工作。

指标定义：time-to-beats-BB、前沿 AUC、novelty yield（§6.3 判）、insight precision、单一化指数（top-k 结构熵）。

## 15. 非阻塞待办

- anchor p 伪阈值自动选点（v1 固定各 profile 默认）；
- slope_offset 重校准执行时点（首批 final_slow 存活者到位后）；
- acquisition_score v1 公式定稿（Q_eff 分位 × d_eff × Δmin_canonical × novelty 距离；版本化）；
- 已知码表数据源清单与更新流程（BB / Mitten / Zoo；依赖 01 稿 GAP 索引转换工具）；
- 学习层协议参数首轮标定（ε、N≥20、bootstrap 口径、ε_arm、并发假设上限）；
- GEPA 简报进化、meta-agent 工具合成、campaign 级学习消融（future work，D7/§6.6）；
- 论文改稿清单（Fig.1 家族列表 bb/hgp/lp_abelian/lp_nonabelian、decoder 列表 Beam+LSD、workflow 段、SEC 段；orchestrator/generator 双 LLM 叙事已与实现对齐）；
- AlphaSyndrome / ASC finalist 对比协议（04 或论文阶段）。

---

## 16. 实施修订（2026-08-16，round 公平性协议 + A5 对齐 + d≥3 门）

用户决策，取代 §9 的 Tsearch 公式与早期 E5/E6 口径：

1. **round 取值**：单点 T 轮实验提取的 per-round 率带 O(1/T) 边界瞬态低估，按 d 取轮数会在跨 d 前沿对比中系统性抬高小 d 码。改为：**搜索/racing/HoF/beats-BB 一律用 campaign 级固定 T_c**（= 该预算点最大可信距离：预算 ≤144 → 8，>144 → 12，与 d_cap 一致；候选与 resident baseline 同 T_c）；**final_slow 维持 T=d**（BB 论文 "Nc=d ... common practice" 口径，对外报数出此处）。`select_memory_rounds` 的 d_hint 依赖移除（fallback = 固定 8）。
2. **per-round 换算对齐论文 A4/A5**：`p_L = 1−(1−p_X)(1−p_Z)`；`ε = 1−(1−p_L)^{1/(kT)}`（ler_definition = "paper_A4_A5_single_T"；旧保真度域 E6 变体废弃，A5 在 [0,1] 全域有定义，无 fidelity-domain 失败模式）。Wilson CI 经同一单调变换直接输出 `per_logical_per_round_ci`；racing 门槛、beats-BB 判定、统计并列分组全部改用该区间（跨轮数可比，无需换算函数）。CSV schema v5 增列。
3. **d≥3 晋级硬门**（`ranking.MIN_PROMOTION_DISTANCE=3`）：d<3 无纠错能力（d=1 不可探测，d=2 仅探测），promotion_queue / all_archive_queue 过滤，HoF reconcile 兜底（含 verified_circuit_distance<3），生成器 RULES 明示。archive 记录保留（同 quarantine 语义）。

## 17. 实施修订（2026-08-26，文档漂移校正）

本节只登记截至 2026-08-26 代码与本稿正文的实际偏差，不改动 §1–§16（定稿）。判定一律以包内源码为准；as-built 形态见 docs/04，权威顺序不变（代码 > 04 > 设计稿）。本节记录事实，不代表任何「不再做」的决定——那属于所有者。

### 17.1 §12 删除与清理清单逐行核对

原表写的是**处置意图**，不是完成记录；正文读起来像已全部执行，实际有两行没走完。逐行对真实目录树核对：

| §12 行 | 状态 | 核对结果 |
|---|---|---|
| postsearch.py / sinter_worker.py / verification/sinter_sweep.py / CLI sinter-* | 已执行 | 四者均不存在，`cli.py` 无 sinter 子命令。该行的迁移门有结论落盘：02 §9.2 第 7 条（passed-with-attribution，残差全部归因于 ldpc 栈 DEM→矩阵转换的列去重） |
| CLI export-stim-* | 已执行 | `export-stim-seed` / `export-stim-queue` 保留（cli.py:392、:404），实现为 `export_stim.py` → `verification/circuit.py:149` → `sec/stim_builder.build_lrc_memory_circuit` 的薄导出。该行给的两条保留理由现只剩一条，见 §17.6 |
| runner.py / orchestrator.py / agents/ / proposal_eval.py / proposal_round.py | 已执行 | 五者均不存在；`orchestrator/` 现为新建的 policy 层包，无重名残留 |
| funsearch.py / campaign.py / slow.py / slow_worker.py | **部分执行** | 只走完一半，见 §17.2 |
| surrogate/ | 已执行 | 目录不存在，`ranking.py:18` 注明 hook-surrogate tie-break 已退役、列名换为 Δmin/τA。残留仅是 `fast_surrogate_*` 这类历史 schema 名（funsearch.py:505、memory/archive.py:671、:1138），不含预测器 |
| README 双语 | 已执行 | `README.md` / `README.zh-CN.md` 均为重写稿，family 列表已是 bb/hgp/lp_abelian/lp_nonabelian，decoder 列表已是 beam B=64 默认 + BP+LSD 保底 |
| PROGRESS.md / E6 状态文档 / hpc/ 旧脚本 | **部分执行** | 文件已移入 `docs/archive_e6/` 与 `hpc/archive_e6/`，但两个归档目录内都没有「归档说明」；全仓唯一的口径声明是 `docs/README.md` 末行「历史 README/PROGRESS 记录仅供参考」。归档目录里的 `README_E6_*.md` 是 E6 期原件，不是说明文 |

### 17.2 funsearch.py / campaign.py 未拆解，且在热路径上

同一行的四个对象只走完一半：`slow.py` / `slow_worker.py` 已消解进 `evaluation/`；**`funsearch.py`（62 KB）与 `campaign.py` 仍是包根模块，且承载 campaign 主链路，不是待清的 legacy**——`workflow/graph.py:348 execute_plan_node` → `campaign.run_agentic_cycle`（campaign.py:46）→ `funsearch.run_funsearch`（funsearch.py:60）；两者由 `__init__.py:21`、`:23` 再导出，`cli.py:25`、`:602` 直接使用。生成器 LLM 的 prompt 也整体留在 `funsearch.py:572 _funsearch_system_prompt` / `:638 _funsearch_user_prompt`，而 §12 的目标形态本应把它放进 `orchestrator/prompts/`——这与 04 §11 R1「generator prompt 不进协议快照」是同一处的两个侧面。**as-built 的分层安排以 docs/04 §1 为准**（04 §1 的分层图把 `funsearch.py` 明列为 fast 环 Generator，但整张图未出现 `campaign.py`；`campaign.py` 的「fast/slow 粘合层」定位见 04 §3 与 §11 R13），另见 04 §11 的残留表——R13 记的正是本行这条未执行的 §12 条目。本稿不判定该拆解是否仍要做：§13 收尾第 10 项「删除清单收尾」在纸面上仍然成立。

### 17.3 §12「仓库最终形态」与 §7 evaluation/ 树是示意草图

两处树都不是落地形态，以 04 §7（产物树）与 04 附「模块速查」为准。已知差异：

1. **无仓库级 `manual/`**。手册是 run 产物：`runs/<campaign>/manual/`，static / dynamic / skills 三区由 `orchestrator/manual.py` 的 `render_static` / `render_dynamic` / `record_skill` 写入（`orchestrator/loop.py:258-264` 每轮渲染），`orchestrator/reflector.py:155` 另写 `manual/dynamic/reflections.jsonl`，读侧是 `tools/queries.py::manual_read`。§2.2 把 `manual_dir` 列为 per-run 字段这一点本身是对的（虽然该键并未接线，见 §17.4），错的只是 §12 树里的位置。
2. `tools/` 实为 `queries.py`（不是 `query_*.py`——八个只读工具合在一个模块）+ `precheck_draft.py` + `acquisition_score.py` + `ledger_ops.py` + `context.py`。
3. `orchestrator/` 另有 `plan.py`（TurnPlan schema 与校验）/ `manual.py` / `reflector.py`，树中未列。
4. 学习层落在独立的 `memory/` 包（`archive.py` / `ledger.py` / `backtester.py` / `bandit.py` / `fingerprints.py` / `known_codes.py` / `coverage.py`），§12 树整包漏列——§6 全靠它。
5. §7 的 `evaluation/decoders/` 与 `evaluation/ler.py` 实际建成 `verification/decoders.py` 与 `verification/ler.py`：decoder 会话封装与 LER 口径这道缝留在了 `verification/` 一侧。`evaluation/` 另有 `baseline.py` / `common.py` / `feedback.py`，树中未列。

### 17.4 §2.2 CampaignState 与 workflow/state.py

实现为 `class CampaignState(TypedDict, total=False)`（state.py:53）。逐项对照：

1. **八个字段声明了但全仓无写无读**：`archive_path` / `queue_path`（:59-60）、`manual_dir` / `ledger_path` / `fingerprint_index_path`（:61-63，源码注 "Milestone B; reserved"）、`bandit_stats`（:66）、`transcript_ptr`（:69）、`rng_cursor`（:70）。因 `total=False`，按 §2.2 直接读这些键会在运行时抛 KeyError（写不会）——不要照 §2.2 写读侧代码。源码里的 "reserved" 也会误导：ledger 与 bandit 功能本身都已实现（graph.py:586、orchestrator/loop.py:264），只是没走这几个 state 键。
2. **实际接线**：产物指针走 §2.2 未列的 `last_cycle["artifact_paths"]`（写于 graph.py:460，读于 :601、:747 与 tools/context.py:38、:53）；manual 与 ledger 路径由 `run_dir` 就地派生（graph.py:721/:735、:586）；RNG 不进 state——种子按 index 派生（campaign.py:386 `seed=0`，evaluation/pool.py:313 `seed + index * 1_000_000`、:784 `base_seed + noise_index * 10_000`），确定性由此保证，本就不需要游标对象。
3. **代码另有四个在用字段而 §2.2 未列**：`last_cycle` / `slow_context_ptr` / `final_slow` / `report`（state.py:72-75）。
4. `hof_summary` 的行注（code_hash / family / plpr@anchor / CI）偏轻：每行由 `_light_slow_row`（graph.py:1183-1211）产出 21 个字段，外加 `spec` 共 22 项——`spec` 是有意入 state 的，`final_slow_node` 要照它重建重测（graph.py:916-919）。
5. `protocol` 行注列的十项里，**`bandit` 不在快照内**：`build_protocol_snapshot`（state.py:130-173）没有任何 bandit 键，`STRATEGY_ARMS` / `EPSILON_ARM=0.05` / `UCB_C=√2` 是 `memory/bandit.py:17-19` 的模块常量，改它们不会被 `--resume` 的整字典比对（checkpoint.py:70）拦下；§8 末行点名要进快照的 ε_arm 同此。其余各项都在：sec_policy_id、decoder 与 noise（经 `verification.to_dict()` + `noise_profile`）、anchor_p、racing 参数、P（= `sec_synthesis.candidates` = 64）、工具预算、max_turns。`T_Δ` 在代码与其余文档中都无对应物，不作判定。
6. `budget` 的「四本账余额」应读作**四本账流水**：`budget` 确实记 turns_used / wallclock_s / tokens_used / slow_shots_used（graph.py:1341-1346），但 `_stop_decision`（graph.py:881-897）唯一的终止条件是 `turns_used >= max_turns`——tokens / slow-shots / 墙钟三本账只记账、无上限判定。同处注释记录了 2026-08-17 用户决策：agent 主动 finalize 与 beats-BB 提前终止一并关闭（Fig.2 前沿要完整迭代曲线；b144 v2 第 2 轮就出现 CI 分离的 beats-BB 码，旧规则会让 60 轮的搜索在第 3 轮收工），campaign 一律跑满 max_turns。§3.2 步骤 7 与 §3.3 的「四本账硬上限」按此修订；`budget_status` 工具返回的 caps 也只有 max_turns 与每轮 tool_budget。

### 17.5 §4 工具体系：注册表实到 12 个，policy 档基本未工具化

`orchestrator/registry.py::registry()` 实际注册 12 个工具，`render_tool_docs()` 只渲染这 12 个（S-1 内联工具目录同源）：

| 工具（真实签名） | scope | 与 §4.2 的差异 |
|---|---|---|
| `manual_read(section="static/INDEX.md")` | read | — |
| `query_archive(family, max_n, min_d_proxy, promoted_only, limit=16)` | read | 是具名过滤参数，不是自由 filter 串 |
| `query_evidence(code_hash)` | read | — |
| `fingerprint_lookup(code_hash)` | read | 参数名非 `x` |
| `known_code_lookup(code_hash)` | read | 参数名非 `x` |
| `acquisition_score(code_hashes)` | read | 参数是 hash 前缀列表，非 `ids` |
| `precheck_draft(program, rng_seed=0)` | draft | — |
| `bandit_status()` / `budget_status()` | read | — |
| `ledger_propose(predicate, claim, scope, rationale)` / `ledger_retract(entry_id)` | policy | policy 档实到的只有这两个 |
| `query_coverage(limit=24)` | read | **§4.2 未列**：按 family / 群类 / 群阶带统计 proposed / measured / censored，并列出从未提案的目录分支 |

签名口径以注册表的 docstring + example 为唯一事实源（§4.1 ③、§5），上表与 §4.2 表都只是说明性的。

§4.2 最后两行不是工具，`@tool` 的四项（schema / 作用域 / docstring→手册 / transcript）对它们都不适用：

- `submit_plan` 是工具环收尾的 JSON 信封键（loop.py:362、:367、:374）。计划体仍受严格校验——`TurnPlan`（plan.py:35-50，`extra="forbid"` + 引用校验）——只是不走注册表的 `create_model` schema；校验失败也不是一次判死：每轮开局把修复预算置为 `format_repair_budget`（默认 4，loop.py:127；构造时 `max(1, …)` 兜底，:135；每轮置位在 :323），不可解析回复、计划校验失败、空工具请求、无可执行键四类共用这一份预算（:356 / :401 / :429 / :456），且每次干净的工具调用回补 1、上限仍为 4（:450），耗尽才转 FallbackPolicy——全仓只有 tests/ 传这个参数，生产口径就是 4。放弃「一轮一次」的理由记在 :319-322 的注释：单次修复会让 `--strict-turns` 下的一次格式失手直接致命。外层另有硬上限——每轮 LLM 调用数 ≤ `tool_budget + 2`（:324），走完即降级。loop.py:6 的 docstring 仍写「one structured repair round」，属过时口径；
- `promote` / `finalize` 实为 `TurnPlan.promotions: list[Promotion]`（plan.py:43）与 `TurnPlan.finalize: bool`（plan.py:50），后者由图边 `continue_or_finalize`（graph.py:900）消费；
- `generate(brief, n)` 不存在。生成由确定性的 `execute_plan_node` 在 turn 收尾后驱动（链路见 §17.2），agent 只能经 TurnPlan 的 `program_count` / `family_mix` / `directives` 间接影响批次（→ `proposal_count`；`_family_quota`（graph.py:496）→ `overlay["family_quota"]`；`_strategy_brief`（graph.py:563）→ `overlay["strategy_brief"]`）。`precheck_draft` 只沙箱试跑 agent 自写的程序，既不触发生成器，也没有注入生成批次的通道。**连带后果**：§3.2 步骤 5「agent 审清单 → promote」在实现中不成立——promotions 随计划在生成之前就定死，且只能引用往轮已知 hash（plan.py:123-135），对本轮新产候选的审阅顺延一个 turn。

受限写的强制点是 `validate_turn_plan`（plan.py:73-167；三项断言分别在 :121-122、:134-139、:140-164）：promote_cap 超限、隔离候选（degraded_* / schedule_catastrophic）被提名、偏离 acquisition 序不带 rationale，都在这里报错回模型，不在注册表的 scope 字段。

**§4.1 的「② 作用域强制」未实现**：`ToolRuntime.call`（registry.py:120-154）依次做预算检查、查名、pydantic 参数校验、派发，全程不读 `spec.scope`；scope 只在 `render_tool_docs`（registry.py:101）用来在手册上打 `[scope=...]` 标注。三档权限当前靠工具集本身的能力边界成立（read/draft 工具物理上没有写协议层的接口），不是运行时断言。

### 17.6 §9 / §13 的 ranked package 未实施

全仓无 ranked package 产物。终局产物是 `report_node`（graph.py:1062-1098）写的 `campaign_report.json` / `campaign_report.md`——含 `final_frontier` top-24、hall_of_fame、baseline、协议快照与四本账流水；`summary.csv` / `progress.csv` 由独立模块事后导出（`python -m omniqec_agentic.summary_csv|progress_csv <run_dir>`），不在图内。§13 步骤 9 的「ranked package 报告」与 §9 末段的 ranked package 均按此理解；§12 保留 export-stim-* 的两条理由随之只剩 qualification set 发布一条。

### 17.7 §10 qualification harness

1. **命令行签名**与 §10 正文示例有三处出入（cli.py:444-469）：选项名是 `--references`（复数），不是 `--reference`；`--out` 必填；`bposd_ref` 不是合法取值——decoder 名只有 beam / lsd / bposd / relaybp（config.py:47-51），高阶 BPOSD 对照通过 bposd 的配置表达。实际形态：`qualify-decoders --set <路径> --out <路径> --references lsd --candidates beam [--profiles uniform si1000] ...`；`--set` 可省，默认 `docs/qualification/qualification_set_v1.json`（qualification.py:37）。
2. **set v1 规模未达标**：`qualification_set_v1.json` 只有 11 码（bb 4 / hgp 3 / lp_abelian 2 / lp_nonabelian 2），且两个 lp_nonabelian 同为 n=24——该族只覆盖单一尺寸，离「4 families × 3 sizes × 2–3 instances ≈ 24–36」还差一截。同句其余四项都成立：四族齐备、BB72/144 带 `anchor: true`、profiles 为 uniform_legacy_v1 + si1000_modern_v1、文件在 `docs/qualification/` 且带 `version: v1`。set 文件自带的 note 已把「扩到 24–36」记为 v2 跟进；harness 本身可用（cli.py:1024 → `qualification.run_decoder_qualification`）。

### 17.8 §16.1 的 T_c 分档未落地

§16.1 写「预算 ≤144 → 8，>144 → 12」；`workflow/graph.py:1300-1305` 全局硬编码 `search_rounds = 12`，注释引同一 2026-08-16 决策并给出改口径的理由：这些预算点的前沿含 d>8 的码，T_c=8 会在 racing 期把它们时间截断；取 12（BB144 d=12 是参考天花板）覆盖本实验的可验距离区间，且对每个候选的边界偏差一致。即分档取消，跨预算点统一取 12。

### 17.9 §0 与 §7 树里的「E5/E6」是旧名

正文两处仍以旧标签指称品质因子：§0 末行「E5/E6 口径、Wilson CI、shard 的测量协议地位」（L39）与 §7 树的 `ler.py  # E5/E6（原样保留）`（L252）。**口径本身按 §16.2 已换成论文的 A4/A5**：`verification/ler.py:1` 的 docstring 即 "Paper-defined logical-error-rate aggregation (SI C, Eqs. A4-A5)"，`:19` 是 `LER_DEFINITION = "paper_A4_A5_single_T"`；全仓 `*.py` 里没有 E5/E6 标识符（`grep -rn 'E5\|E6' --include='*.py'` 唯一命中是 `sec/coloring.py:95` 的哈希常量 `0xBF58476D1CE4E5B9`）。README 双语与 hpc 的两份 Fig.2 平台说明（当时的 NSCC 与 Pod 平台 README，不在开源分支中）都已改写为 A4/A5，docs/04:108、:236 写 `plpr（A5）`。**这两处按 §16.2 读**；L252 的模块地址另按 §17.3 第 5 条改为 `verification/ler.py`。`docs/archive_e6/` 与 `hpc/archive_e6/` 里的 e6 只是归档目录名（§17.1 末行那条），不是仍在用的口径名。

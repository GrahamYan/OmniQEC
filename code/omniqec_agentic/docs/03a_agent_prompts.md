# OmniQEC 设计稿 03a：Agent Prompt 规范（可定位、可溯源优先）

- 状态：**定稿 v1.1**（2026-08-15；随 Milestone B 落地）
- 依据：设计稿 03 定稿 v2.1 + 三路 prompt 实践调研（生成器侧 / 编排器侧 / 溯源工程；来源见 §9）+ 现有代码 prompt 资产盘点（§5）
- 范围：两个 LLM 角色的 prompt 全文与装配规范、prompt 版本化方案、上下文块标注格式、rationale 引用规范、调用记录 schema、现有 quantum prompt 资产迁移清单。
- 设计原则来源：用户指令"按最容易定位、可溯源的方式设计"。
- 修订记录：v1.0 草案 → v1.1 定稿：新增 §5 现有实现资产迁移清单（直迁/升级/废弃三档，用户要求）。

---

## 0. 调研结论摘要（直接塑造本稿的六个事实）

1. **生成器 prompt 的两层解剖是行业标准**（AlphaEvolve → OpenEvolve → ShinkaEvolve → IBM qcode-discovery 一脉相承）：1–3 句通用 system message + 逐轮装配的 user message（父程序 → 紧凑指标行 → 历史/灵感 → 任务与格式规约）。约束一律写成**代码契约**（函数签名/返回类型/标记区），有效性永远由 evaluator 强制——SCE 的反作弊条款原话："All validity checks and scores are computed from these generated matrices, not from textual claims"。这与本框架现有实现（registry 驱动的 format contract、验证器真值）完全同构。
2. **被调研的所有系统中，编排器都不是 persona 化的 LLM**：Google Co-Scientist 的 Supervisor 是统计驱动的任务队列伪代码；Robin 的编排是 Python 代码。persona 只出现在单一职责的 worker 调用上，且一律是**一句话中性专家**（"You are an expert evaluator..."），没有任何论文使用或论证 "lab director" 式人格。被消融验证有效的是 prompt **结构**（辩论式排名降低位置偏差、SWE-agent 的接口式护栏），不是人格。**因此本稿撤回此前"实验室主任 persona"的建议**——我们的 Orchestrator 是 LLM policy agent（03 稿 D1，本身已越出调研样本的做法），prompt 更应保守：中性一句话身份 + 纪律放进工具文档、schema 与工具层断言（SWE-agent ACI 教条：系统 prompt 从 306 词压缩到 14 词，纪律全部迁入接口）。
3. **Rationale-before-verdict 字段序**（Robin 的 {Analysis, Reasoning, Winner, Loser}）：结构化输出里推理字段排在决策字段之前，决策质量与可审计性同时受益。
4. **预算作为一等 prompt 内容**：精确计数指令（"Generate exactly {N}"）、逐轮注入的实时余额（PaperQA 的 cost status）、显式终止路径。
5. **引用纪律的可靠实现是 schema + 校验器拒绝，不是 prompt 恳求**（attribution 研究实证：schema-binding 显著提升 claim precision）；引用只允许上下文中真实存在的 ID，prompt 中给出合法/非法示例文法（PaperQA 模式）。
6. **Prompt 版本化共识**：prompt 是 git 中的不可变文件，内容哈希为身份（`name:hash8`），可移动别名永不进审计记录；每次 LLM 调用落一条最小可复现元组记录。

## 1. 溯源设计的五条硬规则

| # | 规则 | 实现锚点 |
|---|---|---|
| R1 | **Prompt-as-code**：所有 prompt 文本是 `orchestrator/prompts/*.md` 中的版本化文件；`prompt_hash = sha256(规范化文本)`，对外标识 `<name>@<semver>`（人读）+ `<name>:<hash8>`（记录）。修改永不原地——只产生新哈希。`prompt_id` 进 protocol 快照（与 sec_policy_id 同级：**换 prompt = 换协议，跨版本 run 不混比**） | workflow/state.protocol |
| R2 | **上下文块标注来源**：每个注入块包裹 `<source id="S-N" kind="..." origin="<文件/字段路径>" gen="<代数>" hash="sha256:...">…</source>`；每 turn 的块注册表 {source_id → origin, hash} 随 transcript 落盘——任何被引用的 ID 可机械解析回模型当时看到的字节 | orchestrator/loop.py |
| R3 | **Schema 强制引用**：TurnPlan/promote 的 rationale 字段必须携带 `cites: [...]`，仅允许三类 ID——`S-*`（上下文块）、code_hash 前缀、`I-*/H-*`（账本条目）；校验器拒绝未知 ID 或空引用的载荷并要求重出（拒绝在工具层，非 prompt 信任） | tools/registry 校验器 |
| R4 | **数字零转述**：agent 的输出永不成为记录（03 稿 §4.3）；prompt 中明示"你引用数字时必须带其 source id；report 由管道从产物文件渲染" | report 节点 |
| R5 | **调用记录最小元组**：每次 LLM 调用持久化 `{turn, role, prompt_id, prompt_hash, model_id, params, input_hash, output_hash, raw_response_path}`（OTel gen_ai.* 命名兼容）；原始响应先落盘再解析（03 稿 §2.3） | llm/ 客户端 |

## 2. 角色与 persona 定案

- **Orchestrator**：中性一句话身份（非 lab-director，§0-2 的撤回）；全部纪律经三条通道：①工具文档（registry docstring 自动渲染，含每工具一个用例）；②输出 schema（rationale-before-verdict + cites 强制）；③工具层断言（03 稿 §3.3，物理无写接口）。
- **Generative Search Agent**：沿用已实现的 registry 驱动 prompt（01 稿落地），叠加 strategy_brief 槽位与假设标签协议；身份行维持现状风格（"You are the evolutionary engine of OmniQEC..."——一句话、任务绑定）。
- **假设标签载体（定案）**：construct 程序块**首行结构化注释** `# exploits: I-3, I-7; tests: H-12`，由管道解析入档；spec dict 保持纯 genotype。程序无标签合法（视为自由探索）。
- **Rationale 粒度（定案）**：turn 级 rationale 必填（含 cites）；候选级 rationale 仅在**偏离 acquisition 序或 bandit 建议配比**时必填（03 稿 §3.2-5 语义不变）。

## 3. Orchestrator prompt

### 3.1 System prompt（`orchestrator/prompts/orchestrator_system.md`，v1）

```text
You are the campaign policy agent of OmniQEC, a quantum error-correcting
code discovery system. You decide WHAT to try, WHOM to measure, and HOW MUCH
budget to spend. You never decide how things are measured: all measurement
protocols (circuit synthesis, noise, decoding, statistics) are frozen and
physically outside your write access.

## Your decisions each turn (submit via submit_plan / promote / finalize)
- program count and strategy mix for the next generation (within caps),
  including attention allocation across code families {bb, hgp, lp_abelian,
  lp_nonabelian};
- which fast-loop candidates to promote to slow verification, and at which
  budget class (pilot / confirm);
- ledger operations: propose or retract insight entries (a deterministic
  backtester adjudicates them — you have proposal rights only);
- when to finalize the campaign (allowed after min_turns).

## Hard rules
1. Evidence only. Every number you rely on must come from a tool return or a
   tagged context block. When your rationale uses a number or a claim, cite
   its source id. Never state performance figures from memory.
2. Citations. The `cites` field of your plan may only contain ids that exist
   in this turn's context: S-* block ids, candidate code_hash prefixes
   (>= 8 chars), or ledger ids I-*/H-*.
   Valid: {"cites": ["S-3", "I-12", "a41f00b2"]}
   Invalid: citing a paper name, a family name, an id not shown this turn,
   or an empty list on a load-bearing claim.
3. Deviations need reasons. If your promotion list departs from the
   acquisition_score order, or your strategy mix departs from the bandit
   suggestion, state the per-item rationale with cites. Following the
   baseline ordering needs no per-item justification.
4. Quarantine is not yours to lift. Candidates flagged schedule_catastrophic
   or degraded_* are excluded by the tools; do not attempt workarounds.
5. Budget discipline. You see four ledgers (tokens / slow shots / wall clock
   / turns) each turn. Plans exceeding remaining budget are rejected by the
   tools. Prefer finishing with a defensible frontier over exhausting budget.
6. Tool budget: at most {tool_budget} tool calls per turn; the turn is
   force-closed with your current best plan afterwards.

## Output contract
Produce your plan as a single JSON object matching the TurnPlan schema, with
fields in this order: observations (what you learned this turn, with cites),
ledger_ops, rationale (turn-level, with cites), family_mix, strategy_mix,
program_count, promotions [{code_hash, budget_class, rationale?}], directives
(free-text guidance for the generator, e.g. "explore near C3:C4 metacyclic
lifts"), finalize (bool).
Invalid JSON, unknown citation ids, or out-of-cap values are rejected and
returned to you once for repair; a second failure closes the turn with the
fallback plan.
```

要点说明：身份 1 句；规则 6 条全部是"可被工具层验证的行为规则"，无形容词式性格描写；引用文法给正反例（PaperQA 模式）；schema 字段序 = rationale-before-verdict（observations/rationale 在 promotions/finalize 之前）。

### 3.2 每 turn 的上下文装配（`orchestrator_turn` 节点）

按固定顺序注入，全部为 R2 标注块：

```text
<source id="S-1" kind="manual_static_index" origin="manual/static/INDEX.md" hash="...">
  （工具目录 + 模块一页摘要；registry 自动渲染）
<source id="S-2" kind="manual_dynamic" origin="manual/dynamic/" gen="...">
  （supported 洞见全文 + hypothesis 摘要 + bandit 权重快照；ACE 式不摘要化）
<source id="S-3" kind="digest_fast" origin="<run>/digest/gNNN.json#fast_topk">
  （fast top-k：code_hash8 / family / n,k,d_proxy / Q_eff / Δmin_canonical / flags）
<source id="S-4" kind="digest_slow" origin="<run>/digest/gNNN.json#slow_topk">
  （slow top-k：code_hash8 / plpr@anchor + CI / d_circ_ext / stopped_by / 审计标志）
<source id="S-5" kind="hof_baseline" ...>（HoF 摘要 + baseline@anchor）
<source id="S-6" kind="budget" ...>（四本账余额，PaperQA 式实时注入）
<source id="S-7" kind="bandit" ...>（策略臂建议配比 + 置信统计）
<source id="S-8" kind="last_plan_outcome" ...>（上轮 plan 与实际执行差异、被拒项及原因）
```

候选一律以 code_hash 前 8 位指称（提示词与记录同一标识——"最容易定位"的核心：**模型说的 ID 就是产物文件里的 ID**）。

### 3.3 TurnPlan schema（pydantic，校验器拒绝式）

```python
class TurnPlan(BaseModel):
    observations: list[Cited]        # Cited = {text: str, cites: list[SourceId]}
    ledger_ops: list[LedgerOp]       # propose/retract，条目内 predicate 引用已计算字段
    rationale: Cited                 # turn 级，必填
    family_mix: dict[Family, float]  # 和为 1；工具层夹紧
    strategy_mix: dict[Arm, float]   # 偏离 bandit 建议超阈值 → 要求 deviation_rationale
    program_count: int               # <= per-turn cap
    promotions: list[Promotion]      # {code_hash, budget_class, rationale: Cited | None}
    directives: str                  # 传给 generate() 的 strategy_brief 素材
    finalize: bool
```

校验失败 → 一次修复机会（错误以结构化 user turn 返回，Coscientist 纠错模式）→ 再失败即 FallbackPolicy 收 turn（03 稿 §3.4），事件记入 transcript。

## 4. Generative Search Agent prompt

### 4.1 结构（沿用已实现基线 + 三处增量）

System prompt 保持现实现的 registry 驱动装配（身份行、genotype 行、语法示例、STRICT FORMAT CONTRACT、RULES——01 稿已适配四 family；盲测纪律原文保留：永不提及 BB 基准）。增量三处：

1. **strategy_brief 块**（user prompt 首部，R2 标注）：

```text
<source id="S-B1" kind="strategy_brief" origin="turn NNN plan.directives" hash="...">
DIRECTIVES FROM THE CAMPAIGN POLICY:
- attention: {family_mix rendered}
- {plan.directives 原文}
- active insights you may exploit (cite by id in your tags):
  I-12 [supported, scope=lp_nab×n≤200]: <predicate → claim 一句话渲染>
  H-31 [hypothesis]: <...>
</source>
```

2. **假设标签协议**（RULES 追加一条）：

```text
- Optionally tag each program's FIRST line with a comment
  `# exploits: I-<id>[, ...]` and/or `# tests: H-<id>[, ...]` referring ONLY
  to ids listed in the strategy brief. Tags drive credit assignment; untagged
  programs are treated as free exploration. Do not invent ids.
```

3. **反馈词汇表与诊断行**（IBM prompt_context 模式——"教模型读自己的反馈"，进 manual/static/ 由 registry 渲染，不占 system prompt）：每个反馈字段一行语义 + 症状→动作，例如：

```text
schedule_catastrophic: this check structure admits a single-fault logical
  under ALL P sampled schedules — treat as a structural property; avoid
  similar check-overlap patterns rather than resampling parameters.
stopped_by=racing: LER CI fell clearly below both incumbents at anchor p;
  the structure family may still be useful at other budgets.
d_circ_ext < d_proxy: circuit-level distance degradation — inspect the
  residual-heavy checks (high Δ-contributing suffixes) before reusing.
```

### 4.2 反馈块格式（沿行业标准：紧凑标量优先）

每程序一行指标（`[[n,k,d]] Q=... plpr@p=... flags=...`），程序按分升序排（最优最后）；执行错误进独立 fenced 块并截断——现实现已大体如此，迁移时对齐命名即可。

## 5. 现有实现的 quantum prompt 资产迁移清单

现有代码（funsearch.py / slow.py / agents/proposal.py / config.py / registry）沉淀了一批实战校准的领域 prompt 资产。逐条盘点：**直迁**（原文保留）、**升级**（词汇随新证据通道更新）、**废弃**（与新协议冲突）。

### 5.1 直迁（原文进新 generator prompt / manual）

| 资产 | 原文位置 | 去处 |
|---|---|---|
| 身份 + OBJECTIVE："…The cheap first proxy is Q = k*d^2/n, but **the verifier is the source of truth**" | funsearch system prompt | generator system prompt（保留；这句先于 SCE 的反作弊条款表达了同一原则） |
| 反声明规则："Do not claim distance, logical error rate, decoder result, or circuit validity." | funsearch RULES | generator system prompt |
| RNG 纪律："`rng` is a numpy.random.Generator; use it for any randomness so results are seedable." | funsearch RULES | generator system prompt |
| **反平凡变异**："A useful mutation must change the code hash. Do not merely reorder rows, reorder terms, or swap equivalent entries; those can canonicalize to the same code." | agents/proposal.py | generator system prompt（quantum 特有的去重意识，必保） |
| **"Families are construction mechanisms, not search boundaries."** | agents/proposal.py | generator system prompt——正是 03 稿 D3 混合搜索的原生表述 |
| 语法示例故意平庸的原则（"examples anchor LLM proposals, so they must teach the shape without offering a strong starting point or leaking a known reference construction"，Q≈2 的合法平庸码） | funsearch 代码注释 | 升格为 03a 显式教条（进 §2），registry 示例维持此标准 |
| **父代块教条**："TOP PARENT CONSTRUCTIONS … mutate, recombine, or rescale them — **the program is the unit of inheritance, the spec is its best draw**"；源码缩进渲染不加 fence（"so it cannot be confused with the model's own required ```python output blocks"）；baseline 永不入父代池 | funsearch `_parent_construction_lines` | digest/父代渲染原样保留（program-as-genotype 是本框架相对 AlphaEvolve 系 diff-evolution 的立场，必须写明） |
| 预算提升标签："[from a smaller qubit budget]" + scale-up 提示行 | 同上 | 全局库暖启动的渲染（assistant 场景，03 稿 §6.5）——已有实现直接复用 |
| 证据存在才渲染："absent evidence is the norm … printing 'unknown' for every field buries the signal" | funsearch `_optional_evidence_suffix` | digest 渲染规则（S-3/S-4 块） |
| 硬件行：atom/ion all-to-all connectivity 两句 + "Do not propose superconducting-specific assumptions." | config.HARDWARE_PROMPT_PROFILES / proposal.py | generator user prompt 硬件段 |
| REFILL 修复回路（产出块数不足时的结构化补交指令） | funsearch `_refill_funsearch_prompt` | generator 修复回路（与 TurnPlan 修复回路同族） |
| 多 seed 聚合行（"20 seeds -> N distinct codes, M accepted"）、程序结局行（accepted / not better than niche elite / duplicate of / REJECTED(reasons)）、VERIFIER FAILURE ACTIONS（failure_signatures 分组渲染） | funsearch | digest_fast 渲染 |

### 5.2 升级（框架保留，词汇随新证据通道替换）

| 资产 | 升级内容 |
|---|---|
| "MEASURED CIRCUIT-LEVEL RESULTS (**prefer these over Q when choosing parents**)" + 措施行 "Prefer parents with lower measured per_logical_per_round; fall back to Q only for candidates without a measurement." | 证据层级教学原文保留；指标行追加 d_circ_ext / stopped_by / 审计标志 |
| "HOOK-DEGRADED (d_eff=X<d_proxy; **do not clone, repair the cause**)" 与 "These candidates perform worse in the circuit than their code distance suggests — do not clone them: {ids}" | 症状→动作句式保留，标志升级为 `degraded_ext` / `degraded_isd` / `schedule_catastrophic` 三分词汇（02/03 稿），措辞进 §4.1-3 反馈词汇表 |
| slow_feedback 的 `evidence_model` 字典（机器可读的"你如何被排名"说明） | 保留并扩为 manual/static 的排名语义页（IBM"教模型读反馈"模式的本地先例） |
| "High predicted hook risk — restructure rather than copy" | surrogate 退役（03 稿 D12）→ 改为 Δmin_canonical/τA 直接量的同句式（"low Δmin schedules across all P — restructure the check overlap"） |

### 5.3 废弃（与新协议冲突）

| 资产 | 废弃原因 |
|---|---|
| "The run-level deformation mode is fixed to `X`… Never output a deformation field…" | 01 稿定案 deformation 不进 genotype/prompt，LLM 完全不感知——整行删除，不是改写 |
| "Check weight is intentionally never mentioned… the feedback loop is the regulator, not prompt text"（config.py 设计注释） | 该教条已被 01/02 的统一 `w_max ≤ 9` 公平协议取代：weight 上限现在是显式 prompt 契约（registry 行已实现）。注释随 config 清理更新 |
| 旧 JSON proposal 输出契约（agents/proposal.py 的 JSON schema） | legacy 路径随 03 稿 §12 删除；输出协议以 funsearch 的 fenced-code-block 形式为准 |

## 6. Prompt 版本化与调用记录

```text
orchestrator/prompts/
├── orchestrator_system.md      # name=orch-system
├── generator_system.md.j2      # registry 渲染模板（生成器 system 的静态骨架）
├── generator_feedback_vocab.md # 反馈词汇表（manual/static 同源）
└── VERSIONS.md                 # name@semver → hash8 对照表（人读；审计只认 hash）
```

- `protocol.prompt_ids = {"orch-system": "a1b2c3d4", "gen-system": "...", ...}`（run config + 每条 slow_results 行）；
- resume 校验含 prompt_hash 一致性（03 稿 §2.3 的 protocol 校验自然覆盖）；
- transcript 事件采用 PROV-lite 命名：`entity`（prompt 版本/上下文块/plan）、`activity`（llm_call/validate/execute）、`agent`（model_id/policy 名），关系 used / wasGeneratedBy / wasDerivedFrom——查询"哪些晋级决策使用了 I-12"成为一条索引查询。

## 7. 验证与回归（随 Milestone B）

1. prompt 渲染黄金测试：固定 state fixture → 装配文本逐字节比对（含块标注与哈希）；
2. 引用校验器：未知 ID / 空引用 / 越权 ID（协议字段）全部拒绝路径覆盖；
3. 标签解析：首行注释 → 账本信用记账往返；无标签程序不报错；
4. 版本化：改动任一 prompt 文件未 bump hash 对照表 → CI 失败；
5. 修复回路：非法 TurnPlan 一次修复 → 二次失败 FallbackPolicy 收 turn 的端到端用例。

## 8. 待办（非阻塞）

- generator 身份行是否随 orchestrator 定向指令动态化（Shinka 的 5 persona 变体模式）——v1 固定，留消融；
- 反馈词汇表的条目全集（随 evaluation/ 字段定型逐条补）；
- TurnPlan 修复次数（v1=1）与 tool_budget 默认值（v1=32）的标定；
- Co-Scientist 式"meta-review 追加反馈"与 GEPA 简报进化（03 稿二期）的 prompt 接口预留。

## 9. 调研来源（本稿依据）

- 生成器侧：FunSearch（google-deepmind/funsearch）、AlphaEvolve（arXiv:2506.13131）、ShinkaEvolve（SakanaAI 仓库 prompt 模板）、OpenEvolve 默认模板、**IBM qcode-discovery（qiskit-community，逐字 system_message 与 140 行 prompt_context.md）**、**SCE 补充材料（CONSTRUCTION_SPEC 与 mutation cards 逐字）**；
- 编排器侧：Google Co-Scientist 补充材料 Note 8/9（Supervisor 伪代码 + 全部 worker prompt）、Robin prompts.py（835 行逐字）、PaperQA 引用文法、Coscientist (Boiko) SI、ChemCrow prompts.py、SWE-agent ACI 四原则与 14 词 system prompt；
- 溯源工程：LangSmith/MLflow prompt registry 惯例、OTel GenAI 语义约定、Anthropic Citations API 记录结构、schema-binding attribution 研究、W3C PROV / PROV-AGENT、agent-provenance 分层日志综述（arXiv:2606.04990）。

---

## 10. 实施修订（2026-08-26，文档漂移校正）

本节只登记截至 2026-08-26 代码与本稿正文（§0–§9，定稿 v1.1）的实际偏差，不改动正文。判定一律以包内源码为准；as-built 形态见 docs/04 §4，权威顺序不变（代码 > 04 > 设计稿）。本节记录事实，不代表任何「不再做」的决定——那属于所有者。

### 10.1 §3.2 的上下文块实到十个（S-1…S-10）

正文枚举八块。`AgenticPolicy._assemble_blocks`（orchestrator/loop.py:142-201）实际按固定序注入**十块**：S-9（结构覆盖度）与 S-10（近期反思）是后加的两块——04 §10 的两张改动表分别记着它们的加入（「覆盖度仪表 + S-9 块 + `query_coverage`」「Reflector 每轮反思 + 递减窗口 S-10」）。封装格式基本按 R2 落地（`_block`，:91：`<source id kind origin hash="sha256:<前 16 位>">`），但**少一个属性**：R2（§1 L25）与 §3.2 的 S-2 示例都带 `gen="<代数>"`，实现只发 id / kind / origin / hash 四项，全仓没有任何代码给上下文块写 `gen`，落盘的 `context_blocks.json` 行同样只有这四项（:98）。代际因此只能从 turn 目录名与块内容反推，不能从块头直接读。除此之外块登记表每轮落 `turns/turn_NNNN/orchestrator/context_blocks.json`（:273-276），`validate_turn_plan` 认的合法 `S-*` 集合就是这张表的 id 集（loop.py:277 → plan.py:53-70）——R2/R3 的机制本身按设计成立，只是块数、来源与这一个属性与正文不同。逐块实际内容（04 §4 有同源表）：

| id | kind（源码字面） | 实际来源 | 与 §3.2 的差 |
|---|---|---|---|
| S-1 | manual_static_index | `manual/static/INDEX.md` 全文 + `render_tool_docs()`（:156-157） | INDEX 只列小节标题，正文靠 `manual_read` 取；但**注册表 12 个工具的签名/docstring/用例整表内联**（工具清单见 03 §17.5）。理由记在 :150-152：模型不该为了知道工具怎么调而先花一次调用 |
| S-2 | manual_dynamic | `manual/dynamic/insights.md`（:158-159） | 只有 supported 洞见全文 + active 假设（manual.py:90-108）。**bandit 权重快照不在这块**——它是 S-7；另有 `manual/dynamic/bandit.md`（manual.py:110-120）供 `manual_read` 取 |
| S-3 | digest_fast | 上轮 `last_cycle.artifact_paths.all_archive_queue`，缺则回落 `promotion_queue`（tools/context.py:34-48），取前 12 行 | 不是 `<run>/digest/gNNN.json#fast_topk`；渲染字段是 code_hash8 / family / n / k / d_proxy / Q，**没有 Q_eff、Δmin_canonical 与 flags** |
| S-4 | digest_slow | 上轮 `slow_rankings` 去 baseline 行后前 12（:163-167） | 字段 plpr + CI / d_circ_ext / degraded_ext / stopped_by / status |
| S-5 | hof_baseline | `state.hof_summary` + `state.baseline_summary`（:168-171） | 一致 |
| S-6 | budget | `state.budget` + caps（:172-176） | 是流水不是余额，见 §10.2 |
| S-7 | bandit | `bandit.suggestion()`（:177-178） | 一致 |
| S-8 | last_plan_outcome | `state.plan` + `last_cycle.status`（:179-182；status 写于 graph.py:462） | **只有上轮已接受的计划与一个状态串**：「实际执行差异、被拒项及原因」没有进块——计划内的拒因是当轮修复回合里当场回给模型的（loop.py:403），family 配额的请求/产出缺口只写 `turn_audit.json`（04 §11 R15） |
| S-9 | structural_coverage | `coverage_report(archive, slow, limit=16)` + `catalog_gaps()`（memory/coverage.py:27、:90） | §3.2 未列。按区域（family / 群类 / 群阶带）计 proposed / measured / **censored**（racing 在 pilot 深度砍掉、成绩未决，与「测过且输了」分开计），外加 `lp_nonabelian` 四分支计数与 `never_proposed_branches`。:183-186 写明它答的是排名块结构上答不了的问题：零提案区域在 S-3/S-4 不留任何一行 |
| S-10 | recent_reflections | `manual/dynamic/reflections.jsonl`，经 `render_reflections` 递减窗口渲染（reflector.py:184-211；最近 3 条全文、其后至第 10 条各一行 advice、更早丢弃，:41-42） | §3.2 未列。定位是**建议而非证据**（reflector.py 模块 docstring + loop.py:190-193）：定量规律仍须走 `ledger_propose` + 前瞻回测才能被引用。注意这条纪律目前只是散文约定——`validate_turn_plan` 对 `S-10` 与其余 `S-*` 一视同仁，引用它不会被拒 |

### 10.2 §3.2 S-6 与 §3.1 硬规则 5 的「四本账」是流水，不是余额

S-6 载荷是 `{"used": state.budget, "caps": {max_turns, tool_budget}}`（loop.py:172-176），`budget_status` 工具同形（tools/queries.py:171-185）。四本账里只有 turns 有上限判定：`budget` 确实记 turns_used / tokens_used / wallclock_s / slow_shots_used（graph.py:664-665、:452、:458），但 `_stop_decision`（graph.py:881-897）唯一的终止条件是 `turns_used >= max_turns`，campaign 一律跑满（2026-08-17 关闭提前终止的决策记在同处注释）。因此 §3.1 硬规则 5 的第二句「Plans exceeding remaining budget are rejected by the tools」——prompt 文件与 `budget_status` docstring 都在用这句——对 token / slow-shot / 墙钟三本账没有对应实现；计划侧真正会被拒的越界只有 `program_cap` 与 `promote_cap`（plan.py:119-122）。同一处修订见 03 §17.4 第 6 条。

### 10.3 §3.3 的「一次修复机会」实为每轮 4 次、且可回补

§3.3 末行、§3.1 输出契约末句（"returned to you once for repair"）与 §8 待办第三项都写「一次」。实际契约：每轮开局把修复预算置为 `format_repair_budget`（默认 **4**，loop.py:127；构造时 `max(1, …)` 兜底 :135；每轮置位 :323），四类失败共用这一份预算——回复不可解析（:352）、计划校验失败（:396）、空工具请求（:424）、既无 `tool` 也无 `submit_plan` 键（:452）；**每一次干净的工具轮回补 1，上限仍为 4**（:450）；耗尽才转 FallbackPolicy 收 turn。放弃「一轮一次」的理由记在 :319-322：单次修复会让 `--strict-turns` 下的一次格式失手直接致命。外层另有硬上限——**每轮 LLM 调用数 ≤ `tool_budget + 2`**（:324；`tool_budget` 默认 32，workflow/state.py:86，与 §8 待办里的 v1=32 一致），走完即降级。全仓只有 tests/ 传 `format_repair_budget`，生产口径就是 4；`loop.py:6` 的 docstring 与 prompt 文件里的 "once" 都是旧口径。同条另见 03 §17.5 第一条。

### 10.4 §3.3 的 TurnPlan 与实到 schema 的三处差

`orchestrator/plan.py:35-50` 的 `TurnPlan`（`extra="forbid"`）与 §3.3 的字段序一致，差三处：

1. **多一个 `explore_temperature: float`**（plan.py:49，校验区间 [0.0, 5.0]，:116-117）：本轮对当前精英的开采强度，0 保持确定性 top 序，更高值把父代选择放软成 Boltzmann 抽样。它已进每轮 `plan.json`（值出 loop.py:493，落盘 graph.py:339）与 `progress.csv` 的 `explore_temp` 列（progress_csv.py:29、:157），但生产路径上没有调用方把它传给 `archive.parent_records`——见 04 §11 R12，机制存在不等于已接线。
2. **`ledger_ops` 降级为 `list[dict[str, Any]]`，注为 "record only"**（plan.py:38）：真正的账本操作走注册表的 `ledger_propose` / `ledger_retract` 两个 policy 档工具（03 §17.5），计划里的这一栏只是留档。`orchestrator_system.md` 现行输出契约干脆没列这个字段，与 §3.1 引的 v1 文本已不同。
3. **`submit_plan` 是唯一的收尾键**：§3.1 首行写的 "submit via submit_plan / promote / finalize" 里，后两者实为 `TurnPlan.promotions` 与 `TurnPlan.finalize` 字段，不是工具（03 §17.5 第二条）。现行 prompt 文件已改成 "submit via submit_plan"，并新增 §3.1 未收录的 `{"tools": [...]}` 批量信封与 "## How to act" 段。

`Promotion.budget_class`（pilot/confirm）保留在 schema 里，但现行 prompt 自注为 advisory——实际 shot 预算由 racing 控制（03 §8）。

### 10.5 §6 的 prompt 文件树：实到两份 prompt、三个 LLM 角色

`orchestrator/prompts/` 实到 `orchestrator_system.md`、`reflector.md`、`VERSIONS.md`、`__init__.py`。三处差异：

1. **`generator_system.md.j2` 不存在**——生成器的 system 与 user prompt 仍整体留在 `funsearch.py:572 _funsearch_system_prompt` / `:638 _funsearch_user_prompt`（同一件事在 03 §17.2 与 04 §11 R1 各记了一面）。连带后果直接打在 R1 上：`prompt_ids()` 只 glob `orchestrator/prompts/*.md`（prompts/__init__.py:28-38），进 protocol 快照（workflow/state.py:134）的因此只有 `orchestrator_system` 与 `reflector` 两个哈希，**改生成器 prompt 不改协议**——「换 prompt = 换协议」在生成器一侧尚未成立。
2. **`generator_feedback_vocab.md` 不存在**——§4.1-3 的反馈词汇表实现为 `orchestrator/manual.py:34 FEEDBACK_VOCABULARY` 常量，每轮渲染成 `manual/static/feedback_vocab.md`（manual.py:63）并由 INDEX 列出（:69）。落点变了，「进 manual/static、不占 system prompt」的设计意图没变。
3. **多出第三个 LLM 角色 Reflector**（`reflector.md` + `orchestrator/reflector.py:223 run_reflection`），§2 只定了两个角色。它每轮一次、失败即跳过（不失败整轮），调用记录另写 `turn_dir/reflection_call.jsonl` 而不进 `llm_calls.jsonl`，理由记在 :287-289：strict turn 审计查后者认 planner 溯源，咨询性调用不该改变它看到的东西。R5 的最小元组对它是简化版（turn / role / prompt_id / status / error / usage）。

§4.1 的三处增量本身都已落地，只是不在 §6 的树里：S-B1 strategy_brief 块（funsearch.py:660-670，带 sha256 前 16 位，与 R2 同形）、假设标签协议（RULES 行 funsearch.py:619-624，解析在 memory/bandit.py:144 `parse_hypothesis_tags`，另支持 §2 未写的 `strategy: transfer|wildcat` 子句用于 bandit 归因）、反馈词汇表（上条）。

### 10.6 §7-4 的 prompt 版本 CI 门未建

`VERSIONS.md` 的 hash8 列写的是 "(see prompt_ids())"，没有落值；仓内无 CI 配置（无 `.github/`），也没有任何测试引用该文件或 `prompt_ids()`。「改动任一 prompt 文件未 bump hash 对照表 → CI 失败」这道门当前不存在。实际起作用的是另一条路径：`prompt_ids()` 每次现算哈希进快照，`--resume` 对协议快照做整字典比对并列出差异键（workflow/checkpoint.py:60-79）——**改 orchestrator/reflector prompt 会拦下 resume，但不会拦下提交**；改生成器 prompt 两者都拦不下（§10.5 第 1 条）。

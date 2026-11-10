# 搜索流程与论文对齐（os/search，2026-11-10）

原则：论文是规格书。论文写了的，代码照做；论文没写的，保持现状；论文有歧义的，取最字面的读法，并在下文写明。
行号 L 指论文纯文本 `paper14/plain.txt`（双栏抽取，同一行可能夹着另一栏的文字）。

范围：快环检查、种子程序、每轮复盘、编排器上下文（Table I）、迭代计划（Table II）、记忆记录（Table III）。
以下不在本分支范围，未改：快环排序指标（`ranking.py` 的 kd2_v1，Q = k·d²；论文 B2 写 Φ = kd²/n）、晋升门槛（d_search ≥ 5、
结构通道 k ≥ 2、运行时 d/√n ≥ 0.25）、噪声模型、终局 Top-5 补测、参考码、racing 早停（os/measure）、BO/Random/消融（os/baselines）。

## 1. 种子程序只在第一轮执行

| 项 | 内容 |
|---|---|
| 论文 | Methods "Search configurations" L1113–1118："Five manually written seed programs are used to initialize the search during the first iteration, separately from the LLM-generated program budget"；附录 D1 L4229–4234："They are executed only during the first iteration. Subsequent iterations draw on the programs and evaluation records accumulated during the search, without repeating this initialization." |
| 原状 | `workflow/graph.py:execute_plan_node` 调 `run_agentic_cycle` 不传 `use_seeds`（默认 True），每轮 `cycle/fast/programs` 都有 seed_0001–0005。 |
| 改动 | `config.py`：新增 `SEED_PROGRAMS_FIRST_TURN` / `SEED_PROGRAMS_EVERY_TURN`、`seed_programs_policy_of`、`seed_programs_run_this_turn`。`workflow/state.py:build_protocol_snapshot`：新增参数 `seed_programs`（默认 `first_turn`），快照新键 `seed_programs`。`workflow/graph.py:execute_plan_node`：`use_seeds = seed_programs_run_this_turn(protocol, turn)` 传给 `run_agentic_cycle`。`examples/seed_programs.py` 文档串、`docs/final_protocol_20261106.md` 的 Seed programs 一行同步改。 |
| 记忆保留 | 第 0 轮种子产生的结构候选照常进运行级 `structural_ledger.jsonl` 和跨轮 `structural_proxy_elites`（`ArchiveMemory.seed_structural_elites`），实测进名人堂/证据行的也照常保留；后续轮不再重新执行种子。 |
| 歧义 | (a) "first iteration" 取 turn 0。turn 0 中断后从检查点重跑，种子随该轮一起重跑，这仍是第一轮。(b) 没有 `seed_programs` 键的旧快照按旧行为（每轮执行）解释；因为新快照总写这个键，旧检查点在新代码下 `--resume` 会被 `validate_resume_protocol` 拒绝，不会中途悄悄改行为。(c) 论文对 BO/Random 基线没提种子程序；本改动对所有搜索方法一致生效（基线也只在第 0 轮跑种子），基线是否该跑种子交 os/baselines 决定。 |

## 2. 快环合法性检查（两个任务的所有主搜索）

| 项 | 内容 |
|---|---|
| 论文 | Methods L1103–1105、L1166："All main searches require valid CSS codes, at least one logical qubit, a connected Tanner graph and a maximum stabilizer-check weight wmax ≤ 8"；D1 L4225–4226 同句并加 "These constraints apply to both circuit tasks"；B2 L3254–3266：八项检查在距离估计之前（程序能执行并返回合法规格、矩阵二值且维度相容、H_X H_Zᵀ = 0、k > 0、n + m_X + m_Z ≤ N、w_max ≤ 8、Tanner 图连通、最后一项排除已出现过的身份）；"If execution or any check fails, that realization is rejected and its reason is recorded. No replacement seed is drawn"。 |
| 原状 | `verification/structural.py:structural_check_compiled` 已查 CSS 对易、k > 0、总比特预算、连通性（`is_connected_tanner_graph`），也支持权重上限，但 `VerificationConfig.max_check_weight` 默认 None（v3.7 去掉了上限），快环不拦 w ≥ 9；w ≤ 8 只在慢环晋升处（`promotion_max_check_weight`）。另外 `ArchiveMemory._finalize` 先查重、再判结构合法；结构不合法的码也会跑代理距离（distance proxy），而且会登记成 code_hash 的 owner。 |
| 改动 | `config.py`：`SEARCH_MAX_CHECK_WEIGHT = 8`，`VerificationConfig.max_check_weight` 默认改为 8；Fig.4 c 精确权重约束的校验从 "max 必须等于 exact" 放宽为 "max 为 None 或 ≥ exact"（这样 exact ≤ 8 和默认上限能共存；CLI 仍把两者设成同一值）。`memory/archive.py`：`evaluate_and_record` / `evaluate_batch` 只对结构合法且不重复的码跑代理距离；`_finalize` 在查重之前先判结构合法，不合法就以 `structural_reject` 记录（`failure_reasons` 里写原因，如 `max_check_weight_exceeded:9>8`、`disconnected_tanner_graph`），不登记 owner，也不进 MAP 档案、结构池、父代或慢环队列；`_append_attempt` 给结构拒绝行补上结构检查得到的 n/k。并行路径 `evaluate_batch` 原来漏传 `exact_check_weight`（OMNIQEC_FAST_WORKERS > 1 时 Fig.4 c 的精确约束在快环失效），已补。`cli.py`（run-campaign 的 `--max-check-weight`）：不传时为 8；传 0 表示不设上限（v3.7 as-run 协议，属于变体）；传 3 及以上为显式上限，非 8 时打印 PROTOCOL VARIANT。`construction/registry.py` + `funsearch.py:_construction_contract_lines`：有上限时，LP 契约行 "recorded, not capped (v3.7 removed the cap)" 换成 `LP_CAPPED_WEIGHT_LINE`，不再与 CHECK WEIGHT CAP 行自相矛盾（docs/fig3_prompt_map_and_hold_20261022.md 第 4 条记过这个矛盾）。`workflow/state.py`：CampaignSpec.weight_limit 注释。慢环 w ≤ 8 门槛保留，作为冗余保护。**2026-11-10 合并后补充（用户定：w_max ≤ 8 对 Fig.2、Fig.3 一律适用）**：`cli.py` 不传 `--promotion-max-check-weight` 时，慢环门槛默认等于快环上限（默认 8；`--max-check-weight 0` 时不设门槛），快照两处都写明 8；`construction/registry.py` 给 LLM 的 BB 契约行由 "len(A)+len(B) ∈ [4, 9]" 改为 "[4, 8]"（BB 的校验权重就是 len(A)+len(B)），构造器 `construction/bb.py` 的 4..9 语法上限不变，以便重建历史上 w = 9 的码。 |
| 交界 | `hpc/validate_fig2_campaign.py:_validate_protocol`（os/measure 的区域）：final 档（fig2_final / fig3_final）期望 `verification.max_check_weight == 8`，v3_7 / v3_8 档仍期望 None（它们核对 as-run 数据）。只改了这一处断言和一个常量。 |
| 歧义 | (a) "binary and compatible dimensions"：各编译器对矩阵取 mod 2，HGP 种子另有 0/1 语法检查（`construction/grammars/classical_seed.py`），维度由 `shape_mismatch` / `malformed_matrix` 检查，均保持现状。(b) "excludes an identity already encountered in the search"：字面可以理解为整个发现运行，但代码只在单轮快环内按 code_hash 查重（跨轮靠运行级结构账本按 hash 去重、慢环排除已测码和孪生码）。改成跨轮查重会改变 MAP/运行时通道的候选来源，属于设计决定，**未改**，见第 5 节。只把顺序改成"合法性检查在前，查重在后"：同一个不合法的码第二次出现时，仍按 structural_reject 拒绝并写明原因，而不是记成 duplicate。(c) Fig.4 c 精确权重搜索保持 `exact_check_weight` 约束（`max_check_weight = exact`）。(d) 消融、BO/Random 基线都走同一个 `run_funsearch` → `ArchiveMemory.evaluate_batch` → `construct_stage` 路径（基线的提案由 `_baseline_sampler_batch` 生成后进同一个 batch），所以同样受这条约束；基线的慢环选择（`baselines/selection.py`）从结构账本取候选，被拒的码根本进不了账本。 |

## 3. 复盘（reflector）每轮都要有

| 项 | 内容 |
|---|---|
| 论文 | B4 L3410–3413："At the end of each iteration, after the slow-loop measurements have been collected, the same LLM backend is called to review the completed search. It receives the iteration plan, the generated constructions, the screening results and the new circuit-level measurements." |
| 原状 | `orchestrator/reflector.py:run_reflection` 只调一次；客户端自己有 3 次短退避重试，用完仍失败就写 status=failed 并跳过。归档运行里 2236 轮中 78 轮失败（auth_unavailable 503、525/1016 网关错误、超时、429、402），另有 17 轮回复里没有 JSON，却记成 ok 的空复盘。 |
| 改动 | `orchestrator/reflector.py`：`REFLECTION_ATTEMPTS = 3`，退避 `REFLECTION_BACKOFF_S = (60, 300)` 秒，`REFLECTION_WALL_BUDGET_S = 1800` 秒（超过后不再开新的尝试），`reflection_policy()`；`run_reflection` 改成有界重试，每次尝试在 `reflection_call.jsonl` 写一行（含 `attempt` / `max_attempts`）；回复里没有 JSON 对象算失败（`UnparseableReflection`），会重试；最终记录带 `status`、`attempts`、`policy_id`，失败时带 `error`。`sleep` / `clock` 可注入，供测试用。`workflow/state.py:build_protocol_snapshot`：快照新键 `reflection_policy`。`workflow/graph.py`：`_run_turn_reflection` 返回结果（ok / failed / not_applicable）；新增 `_record_reflection_audit`，strict 运行写进 `turn_audit.json` 的 `reflection` 键；`_validate_all_turn_audits` 汇总里新增 `reflection_turns` 和 `reflection_failed_turns`，会进报告的 `turn_audit`。复盘失败只记录，不让该轮失败。 |
| 歧义 | (a) "each iteration" 不包括没有 LLM 的情形：dry_run、BO/Random 基线、no_feedback 消融（论文 Table VIII 中它没有反馈通道）。这几种在审计里记 `not_applicable`。(b) 重试次数和退避时长论文没给，取有界的小值，写进协议快照。 |
| 复盘输入 | 同一句话里的 "the screening results"：原来 digest 只有计划、快环前几名和实测，没有快环筛选结果。新增 `workflow/graph.py:_with_fast_screening`：把本轮 `funsearch_summary.json` 的接受/拒绝/重复计数、最常见的 8 种拒绝原因和产出族计数，放进 `reflection_input` 的顶层块 `fast_loop_screening`（`orchestrator/reflector.py:reflection_input`）。 |

## 4. 附录 B 表格与流程逐项比对

| 论文 | 代码 | 结论 |
|---|---|---|
| Table I Recent outcomes：上一轮计划、近期评估结果、复盘（短期） | S-8 last_plan_outcome、S-4 digest_slow、S-10 recent_reflections（`orchestrator/loop.py:_assemble_blocks`） | 一致 |
| Table I Structural archive：保留的构造、参数、校验权重、程序记录 | S-3（code_hash、family、n、n_total、k、d_search、max_check_weight、pareto_rank）；程序记录可用工具查 | 一致 |
| Table I Measured archive：电路级错误率、不确定度、领先码 | S-5 名人堂 + 参考码；S-4 含 CI | 一致 |
| Table I Search guidance：族覆盖、**剩余评估容量**、策略建议 | S-9 覆盖、S-7 bandit；S-6 原来只有已用量和上限，没有剩余量 | **已改**：`orchestrator/loop.py:_remaining_capacity`，S-6 加 `remaining`（含本轮在内的剩余轮数、每轮慢环槽数、计划可点名的运行时槽数） |
| Table I Hypothesis list：提出的关系**及其内部筛查状态**（持久） | S-2 insights.md 原来只列 supported 和 active，refuted 不出现 | **已改**：`orchestrator/manual.py:render_dynamic` 增加 "Refuted" 一节（最新 12 条，每行带 `[refuted …]` 状态）；graveyard 规则不变，不能再次提出 |
| Table II family mix → 换算成各族程序数 | `graph._family_quota` → overlay `family_quota`，审计记录产出 | 一致 |
| Table II strategy mix → 写进 code-search prompt | `graph._strategy_brief` "strategy emphasis" | 一致 |
| Table II program count → 按固定生成预算检查 | `plan.validate_turn_plan` program_cap + `fixed_program_count` | 一致 |
| Table II promotions → 在指定慢环槽内考虑 | 占用运行时槽，不额外加预算 | 一致 |
| B1 计划执行前检查族/策略/分配/候选 id，不合格退回修改 | `validate_turn_plan` + 修复轮（`AgenticPolicy.plan`） | 一致 |
| B1/B4 复盘滚动窗口，旧复盘缩短后移出 | `render_reflections`（3 条全文 + 共 10 条） | 一致 |
| B4 Eq. (B4) 策略权重 | `memory/bandit.py`：每轮每个臂各拉一次，UCB 加成对所有臂相同，减去最小值后抵消，等价于 v = r̄ − min r̄ + δ（δ = 1e-9），ε = 0.05；初始均匀；奖励按新入名人堂候选平分 | 一致 |
| B4 "These weights inform the orchestrator's next iteration plan πt+1 and reach the code-search agent through its prompt" | 原来权重只给编排器（S-7），生成器只看到计划里的 strategy_mix | **已改**（取字面读法：权重本身进入生成器 prompt）：`graph._strategy_weights_line`，在策略简报里加 "strategy recommendation weights … recommendations, not quotas"；no_feedback 下简报为空，不受影响 |
| B4 假设筛查：只用提出之后的测量，均值 log LER 差 + bootstrap → supported/refuted/保持 | `memory/backtester.py` | 一致 |
| Table III Program / Candidate / Measurement / Iteration / Hypothesis / Strategy statistics | programs/*.py + 首行标签；结构账本行（program_id、rng_seed、spec、code_hash、canonical_hash、快环属性）；evidence_rows / slow rankings；turn_plan.json + digest.json + reflection.json；ledger.json；bandit.json | 一致（Candidate 只收结构合格且 d_search ≥ 5 的实现，门槛问题等用户定） |
| D1 编排器每轮最多 32 次工具调用；规划/生成输出上限 64 000，复盘 8 000；单次请求 900 s；程序不足时可补调用 | `tool_budget = 32`；Claude/OpenAI/OpenRouter 客户端封顶 64 000；`REFLECTION_MAX_TOKENS = 8000`；`LLMConfig.timeout_s = 900`；`MAX_COMPLETION_ATTEMPTS = 3` | 一致（直连 DeepSeek 客户端 `llm/deepseek.py` 用 `LLMConfig.max_tokens = 131072`；论文的 DeepSeek 臂走 OpenRouter，封顶 64 000，故未改） |

## 5. 需要用户决定、本分支未改的出入

1. **快环跨轮查重**（B2 L3259："excludes an identity already encountered in the search"）：现在只在单轮内按 code_hash 查重。若改成整个运行内查重，前几轮出现过、但没测过的码在后续轮里就不再进 MAP 档案和运行时通道（只能经结构精英再被选到），会改变慢环选码，需要设计决定。
2. **计划中论文没列的字段**：`explore_temperature`（只记录，搜索不读取）、`directives`、`observations`、`finalize`（提前结束已禁用）保持现状。
3. 用户指定不动的：Φ = kd²/n 与 Q = kd²、晋升门槛、噪声模型。

## 6. 对运行和其他分支的影响

- 协议快照新增 `seed_programs`、`reflection_policy` 两个键，`verification.max_check_weight` 变为 8，生成器 prompt 多一行 CHECK WEIGHT CAP，LP 契约行改写，S-6 多一个 `remaining`。release/final 的旧检查点在本代码下不能 `--resume`（会报协议快照不一致），旧运行要续跑或重进终局，须用对应的旧代码标签。
- S3 消融的 golden（`tests/golden/s3_ablation/`）用 `_regen_s3_goldens.py` 重新生成；差异只有上面这些。
- os/measure：改了 `hpc/validate_fig2_campaign.py` 一处（final 档期望快环上限 8）。`docs/final_protocol_20261106.md` 改了两行（种子、校验权重），加了一行（复盘）。turn_audit 和审计汇总新增的键是信息性的，验证器不检查。
- os/baselines：基线和主搜索共用快环路径，所以 w ≤ 8 和"种子只在第 0 轮"都对基线生效；论文没说基线要不要跑种子，由该分支决定是否对基线关掉种子（例如在快照里写 `seed_programs` 或在 `execute_plan_node` 按 search_method 分支）。

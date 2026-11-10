# S3 消融实验：反馈路径审计、两种消融的实现与运行手册（2026-10-23）

分支 `s3-ablation`（基线标签 `exp/fig3-cnot-v2` = 3a81c62）。附录方案
`output/research_context/appendix_S1_S4_experiment_plan_20261023.txt` 第 4 节：三组 = Full（OmniQEC）、
去 slow（w/o slow，fast-loop-only）、去 slow+fast（w/o slow & fast，no-feedback），backend Claude，
Memory（Fig.2 v37 流程）与 CNOT（Fig.3 formal_v2 流程）× N98/180/260，Full 复用 6 次历史运行，两个消融组各跑 1 次，
每点 bar 取终局 Top-3 均值。

代码开关：`run-campaign --ablation {full,no_slow,no_feedback}`（`config.ABLATION_IDS`），启动脚本 env
`ABLATION=`；记录在协议快照键 `ablation`（Full 不写该键，快照逐字节不变）；生成器 `feedback_mode` 随之为
`fast` / `none`（`config.ABLATION_FEEDBACK_MODE`）。

## 1. Full 流程里的全部反馈路径（实现级审计）

每轮 t 的顺序：orchestrator 规划 → 生成器 4 代 × 24 程序（fast 环：构造、合法性、结构代理 d_search、MAP-Elites 归档）→
慢环（slow 队列 2S+4R / 3S+4R，电路级测量）→ reconcile（HOF、结构前沿、学习层）→ reflector。

### 1.1 fast 环（结构层）反馈路径

| 编号 | 路径 | 入口 → 出口（代码） | 谁看到 |
|---|---|---|---|
| F1 | 归档表 RETAINED CONSTRUCTION CONTEXT | `memory.archive_summary` → `funsearch._funsearch_user_prompt`（`show_fast`） | 生成器 |
| F2 | 最近程序结果 YOUR LAST N PROGRAMS（含 d_search 拒绝、保留/被占等结论） | `memory.recent_program_summaries` → `_program_summary_line` | 生成器 |
| F3 | 构造失败签名 VERIFIER FAILURE ACTIONS | `memory.failure_signatures` → `_failure_action_lines` | 生成器 |
| F4 | 结构层 FEEDBACK GUIDANCE | `memory.guidance` | 生成器 |
| F5 | 结构父代（跨轮结构前沿 6S + 多样性 4D）SELECTED PARENT CONSTRUCTIONS | `graph._machine_parent_specs` → overlay `next_parent_specs` → `_parent_construction_lines`（`show_memory`） | 生成器 |
| F6 | 代内反馈：第 g 代看到第 g-1 代的归档/最近程序（同 F1/F2） | `run_funsearch` 每代重建 `prompt_context` | 生成器 |
| F7 | 规划器输入：结构前沿 S-3、覆盖 S-9、bandit S-7、手册 S-1/S-2；规划器输出 family_mix/strategy_mix → S-B1 STRATEGY CONTEXT 与 Required family allocation | `loop._assemble_blocks`；`graph._strategy_brief`、`_family_quota` | orchestrator → 生成器 |
| F8 | reflector 的结构输入（structural_frontier、fast_top_k）→ S-10 建议 | `reflector.reflection_input` → `loop` S-10 | reflector → orchestrator |
| F9 | GENERATION: n 计数行 | `_funsearch_user_prompt`（`mode != "none"`） | 生成器 |

### 1.2 slow 环（运行时层）反馈路径

| 编号 | 路径 | 入口 → 出口（代码） | 谁看到 |
|---|---|---|---|
| S1 | 本轮慢环结果 → `prompt_context_slow_aware.json` → RUNTIME FACTS / BPOSD RUNTIME FACTS、"completed circuit-level measurement" guidance | `campaign._run_slow_stage` → `graph.execute_plan_node` 读 `slow_context_ptr` → `_slow_feedback_prompt_lines` | 生成器 |
| S2 | 累计证据行 `evidence_rows.jsonl` → MEASUREMENT FACTS BY CHECK WEIGHT / SYMMETRY、NOT COMPETITIVE（Fig.3）；同一列表喂 `ArchiveMemory(evidence_rows)` 的 expected_plpr 运行时排序（慢队列 R 名额与 promotion_queue 顺序） | `graph._measured_evidence_rows`、`_evidence_tables`；`ranking.archive_order_key` | 生成器；机器 |
| S3 | 名人堂 `hof_summary`（reconcile 从慢环排名建）→ 运行时父代 6R（父代行带 plpr）、S-5、reflector 的 slow_top_k、终局 4 个 runtime 名额 | `graph.reconcile_node`、`_runtime_parent_specs`、`_final_entries` | 生成器、orchestrator、reflector、终局 |
| S4 | 已测码身份 + 上一轮慢队列摘要（Fig.3 v2）→ YOUR LAST N PROGRAMS 的 ALREADY MEASURED / TWIN 标注、LAST TURN'S SLOW QUEUE | `_measured_identities`、`_last_turn_queue_summary` → `_measured_lookup`、`_last_turn_queue_lines` | 生成器 |
| S5 | orchestrator 的 S-4（慢环排名）、S-5（HOF/baseline）、S-9（覆盖含 slow rows）、工具 acquisition_score 等；规划器点名 promotions → 慢队列 R 名额 | `loop._assemble_blocks`、`tools/*`、`campaign._build_slow_queue(priority=...)` | orchestrator；机器 |
| S6 | 学习层：证据行 → backtester 裁决 ledger → STRATEGY CONTEXT 的 adjudicated insights；新 HOF → bandit 奖励 → S-7 | `graph._reconcile_learning`、`_strategy_brief` | 生成器、orchestrator |
| S7 | 驻留参考码 BB 在第 0 轮慢环测一次 → S-5 baseline、终局 baseline 行 | `campaign.run_agentic_cycle(include_baseline)` | orchestrator、终局 |
| S8 | 不可测/解码不可行哈希（慢环结果）→ 从慢队列与终局剔除 | `reconcile_node` unmeasurable/decode_infeasible | 机器 |
| S9 | reflector 的运行时输入（measured_this_turn、slow_queue、baseline、task_evidence）→ S-10 | `reflector.reflection_input` | reflector → orchestrator |

## 2. 两种消融各切断什么

| 路径 | Full | 去 slow（no_slow） | 去 slow+fast（no_feedback） |
|---|---|---|---|
| F1–F4、F6、F9 生成器结构块 | 有 | 有（`feedback_mode=fast`） | 无（`feedback_mode=none`：归档/最近程序/失败/guidance/GENERATION 全部不渲染） |
| F5 父代 | 6S + 6R + 4D = 16 | **12S + 4D**（R 名额归结构池，窗口仍 16；`_machine_parent_specs(ablation="no_slow")`） | **无**（overlay `next_parent_specs=[]` 且 authoritative，归档自带父代也被替换） |
| F7 规划器 | AgenticPolicy（LLM 工具环） | AgenticPolicy，上下文首块加 S-0 ABLATION NOTICE（告知本轮无电路级测量、promotions 无效）；S-B1/家族配额照常 | **NoFeedbackPolicy**（`orchestrator/policies/ablation.py`，无 LLM 调用、无家族配额、无 promotions、无 ledger 操作）；S-B1 与配额不渲染 |
| F8/S9 reflector | 有 | 有（只看到结构记录，slow 输入为空） | **无**（`reconcile_node` 跳过） |
| S1 慢环与 slow overlay | 每轮 | **不跑慢环**（`campaign.run_agentic_cycle(ablation)` 不建队列、不测；status 仍为 ok）；`execute_plan_node` 不读 `slow_context_ptr` | 同左 |
| S2 证据行/事实表/运行时排序 | 有 | **不计算**（`evidence_rows=[]`，`_evidence_tables` 不调用） | 同左 |
| S3 HOF/运行时父代/终局 R 名额 | 有 | HOF 恒空；终局 **6 个结构名额**（见 §3.3） | 同左 |
| S4 已测身份/上轮队列 | Fig.3 有 | **不计算** | 同左 |
| S5 S-4/S-5/promotions | 有 | S-4/S-5 空（由构造而非偶然：无慢环产物）；promotions 被忽略（`promotion_priority=None`） | 无规划器 |
| S6 学习层 | 有 | 机器照常运行但证据为空（backtester 无行、bandit 零奖励）；ledger 仍可由规划器提出结构假设 | 机器照常、无人读取 |
| S7 baseline | 第 0 轮测 | **不测**（终局无 baseline 行） | 同左 |
| S8 不可测剔除 | 有 | 搜索期不存在；终局阶段的解码预算规则照常 | 同左 |
| 严格审计 | plan policy = agentic_v1，须有 orchestrator 调用记录 | 同 Full | plan policy = `ablation_no_feedback_v1`，**不得**有 orchestrator 调用记录（`config.ABLATION_PLAN_POLICIES`） |

机器侧照常但对 LLM 不可见的部分（两种消融相同）：构造合法性/资源检查（能构造、CSS 合法、`N_data+N_checks ≤ N`、
k ≥ 1、CNOT 构造期权重上限 6）、结构代理 d_search（4 种子 × 120 次）、MAP-Elites 归档、跨轮结构前沿
`structural_ledger.jsonl` → `_reduce_elites_from_store`、d_search ≥ 5 与 k ≥ 2 的结构门槛、5 个内置种子程序进入结构池
（与 Full 相同）。no_feedback 的"搜索结束后对全部合法不同码一次性算结构代理"在实现上等价于逐轮算（代理对同一码
确定、前沿满足 elites == reduce(everything ever)），只是算得早；没有任何一步把它反馈给 LLM。

## 3. 本次冻结的执行规则（方案 a–d 的未定项，可改）

### 3.1 去 slow（fast-only）
搜索期不跑慢环、不产生 LER。生成器只收到结构层（F1–F6、F9）与规划器的结构性指令（F7）；reflector 只看结构记录。
父代 12S + 4D。终局名单 = 6 个（与 Full 的 2S+4R 同名额），全部按结构通道自己的名额顺序取：高 k 端点、高 d_search 端点、
（CNOT 还有）最轻格子端点、然后按唯一格子的拥挤代表（`ranking.select_structural_slots(..., slots=6, lightest=profile)`），
25k/基、关 racing、T 按任务（Memory max(12,d_search)、CNOT 固定 12）。

### 3.2 去 slow+fast（no-feedback）
每轮 4 次独立生成器调用（对齐 4 代 × 24 程序），4 个 user prompt 逐字节相同（无 GENERATION 行、无任何历史），
system prompt 与 Full 相同（含任务句）。程序照常 40 种子构造，只做基础合法性/资源检查；不做任何排名反馈。
搜索结束后按 §3.1 同样规则从结构前沿取 6 个进终局——这是事后筛选，不是反馈。无规划器、无 reflector。

### 3.3 终局与 bar
两组终局测量预算相同（25k/基，X/Z 两基，128 分片，终局 RNG 命名空间不变）。每点 bar = 终局有效候选按
(LER 点估计, CI 上界) 排序取前 3 名 LER 的算术平均；有效候选不足 3 个如实记录。不注入 Full 的冠军、候选库或记忆。
终局不再有 baseline 行（搜索期未测参考码）。

### 3.4 其它
轮数 30、提案机会 24×4×40、模型/温度/token/timeout 与对应 Full 一致（Memory `claude-opus-5`，CNOT `claude-opus-5(medium)`，
温度 1.0、64000 token、900 s），成本按真实 token/CPU/时间记录（progress.csv、launcher 日志）。128 核节点（与 Full 相同；
no_feedback 搜索期只有 LLM + fast 环，本可用小节点，但正式运行不试未验证的节点配置，见回报）。

## 4. 协议快照 diff：消融运行 vs 对应 Full

工具：`python omniqec_agentic/hpc/s3_ablation_protocol_diff.py <消融 protocol_snapshot.json> <Full 的>`，
把每个差异分类为 ABLATION / LAUNCH / FIG3_SWITCH / PROMPT_IDS / DIFF，有 DIFF 即失败。

- **CNOT 组 vs Fig.3 formal_v2 Claude 运行**：本机 smoke（b144）除 smoke 自身缩小的规模键外，verification、
  structural_retention、racing、budget_policy、decoder、SEC、noise、prompt_ids 全部一致；正式运行只多 `ablation` 与
  `feedback_mode`。
- **Memory 组 vs Fig.2 v37 Claude 运行**：除上述两键外还有三类已知差异（这棵树是 Fig.3 血统）：
  1. FIG3_SWITCH：`lightest_endpoint_slot=False`、`measured_twin_feedback=False`、`not_competitive_*`、`verification.task=memory`、
     `verification.rounds_policy=max_Tc_d` 在 v37 快照里不存在、这里以正式默认值出现（验证器把缺失读作正式值）。
  2. PROMPT_IDS：`orchestrator_system.md`（2c8937a2 → 12a32b42）与 `reflector.md`（6ebdb483 → 00855f2e）在 v3.7.1 改了
     措辞（三轴前沿、2+4 名额、权重已计入测量值；v37 原文还写着"four of the six slow slots"的旧文字）。只影响 no_slow 组
     （no_feedback 没有规划器/reflector）。
  3. 生成器 prompt 的 v3.7.1 事实列：RETAINED CONSTRUCTION CONTEXT 多一列 w，YOUR LAST N PROGRAMS 带 `w=`；
     Fig.2 v37 Full 没有。只影响 no_slow 组。
  4. **v3.7.1 跨轮前沿归并保留权重轴**（`graph._STRUCTURAL_LIGHT_FIELDS` 含 max_check_weight；v37 Full 在 reconcile 时丢掉
     权重轴，跨轮前沿退化成 (k, d_search) 前沿）。在消融组里结构前沿是父代与终局的唯一来源，所以这是 Memory 消融组与
     Memory Full 之间最实质的保留规则差异（CNOT Full 本就带此修复）。
- 启动键：`slow_workers`（LAUNCH）；racing 策略（搜索期慢环不跑，终局本就固定发数，不影响测量）。

## 5. 运行与运维

当时的部署记录（集群账号、项目号、代码树与输出路径、链表、运维脚本 `s3_ablation_env.sh` / submit / launch / rechain /
pause / credit guard / audit / probe / watch 与快照核对脚本）是账号本地文件，不在开源分支中。保留在这里的是与代码有关的部分：

- 输出：`<OUT_ROOT>/fig2_claudesub_b<N>_<RUN_TAG>`，当时的 RUN_TAG = `s3abl_no_slow` / `s3abl_no_feedback`（Memory）、
  `f3cnot_s3abl_no_slow` / `f3cnot_s3abl_no_feedback`（CNOT）。
- 启动脚本 `nscc_claude_grid_v36.pbs` 新增 `ABLATION=` 环境变量（协议变体：VALIDATE_FORMAL=0、RUN_TAG≠v37），
  当时其余 Memory/CNOT 的 env 与 Fig.2 v37 / Fig.3 formal_v2 启动完全相同（CNOT：TASK=cnot_dense、ROUNDS_POLICY=fixed_Tc、
  PROMOTION_MAX_CHECK_WEIGHT=6、MAX_CHECK_WEIGHT_HARD=6）。
- 论文配置（Table VIII）在预设上运行：`PRESET=fig2_final|fig3_final ABLATION=no_slow|no_feedback VALIDATE_FORMAL=0
  POINTS="98 180 260" PROJ=<PROJECT> bash hpc/submit_claude_v36_grid.sh`（见 `docs/paper_alignment_baselines.md`）。

## 6. 测试与 smoke

- `tests/test_s3_ablation.py`（30 条）：开关关闭时生成器 prompt（both/fast/none × 两任务）、orchestrator 上下文块、
  两任务协议快照与标签态的 golden（`tests/golden/s3_ablation/`，`_regen_s3_goldens.py` 在标签态生成；both/fast、
  orchestrator、快照的 md5 与标签态完全一致，none 组的 golden 因去掉 S-B1/家族配额而重生成）逐字节相同；
  no_slow / no_feedback 各自切断的路径；NoFeedbackPolicy；严格审计；终局 6 结构名额；CLI 与启动脚本接线。
- keyless smoke（`--provider dry_run`，b144，2 轮 × 2 代 × 2 程序 × 2 种子，终局 200 发）：
  `--ablation no_slow`（Memory）与 `--ablation no_feedback --task cnot_dense --rounds-policy fixed_Tc --max-check-weight 6
  --promotion-max-check-weight 6`（CNOT）均跑通到终局（快照带 ablation 键；轮内无 slow 目录；no_feedback 每轮两代 prompt 哈希相同、
  无 orchestrator/reflection；终局 6/0 名额、1 个结构候选测得 LER）。

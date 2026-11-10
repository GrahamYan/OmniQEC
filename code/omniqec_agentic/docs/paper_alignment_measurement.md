# 测量与终局：与论文对齐（2026-11-10，分支 `os/measure`）

规格书是论文（`paper14/plain.txt` 的行号记为 L，`paper.txt` 版面行号与论文自带行号一并给出）。原则：论文怎么写，代码就怎么做；论文没写的保持现状；论文有歧义时取最字面的读法，并在下面写明。

新规则都起了新 id，旧运行不会被误认：

| 规则 | 新 id | 旧 id |
|---|---|---|
| 搜索期早停 | `paper_d3_wilson10x_ref_maxerr30_v1`（`config.paper_racing`） | `pilot10x_maxerr30_v3` / `pilot10x_maxerr100_v3` |
| 终局 | `paper_top5_cumulative_v1`（协议快照键 `final_stage`，`evaluation/final_stage.py`） | 快照无 `final_stage` 键 = 2 结构 + 4 运行时、25 000 发，外加独立工具 `hpc/top5_extension.py` |
| 轮数 | `fixed_Tc`（R = 12） | `max_Tc_d`（T = max(12, d_search)） |
| Fig.4 c 固定权重搜索的终局（11-10 论文修订，§8） | `paper_fig4c_top6_confirm_v1` | 同主搜索 `paper_top5_cumulative_v1` |

**2026-11-10 论文修订**：参考码的报告发数（50 000 / 100 000）和 Fig.4 c 的终局（6 个、只做初始确认）已按新论文改，见 §8；§1、§2 中被取代的句子保留原文，并标了"→ §8"。

---

## 1. 终局确认 = 每个运行的前 5 名、25 000 发 + 补到 Table VII 累计上限

**论文原文**
- D3，L4485–4492（论文行 1954–1960）："After the final iteration, the successfully evaluated candidates are ranked using their search-stage LER estimates, and the five leading distinct codes from each discovery run are selected for final confirmation. These finalists receive additional independent sampling under the same circuit and decoder configuration. An initial confirmation measurement uses 25,000 shots per basis, followed by further shots to reach the budget-dependent cumulative limits in Table VII. Existing completed measurements and new shots are combined by adding the shot counts and block failures separately for each basis. The resulting LER estimates determine the final ranking and the champion of the run."
- Table VII，L4496（`paper.txt` 3032 行）："Cumulative shot limits per measurement basis for top-five final confirmation."
- 正文 L1175–1192（论文行 787–793）；B3 L3344–3346（论文行 1643–1646）；C4 L4170（"Final confirmation uses a larger fixed shot budget"）。

**改了什么**
- `config.py`：`FINAL_STAGE_PAPER_ID`、`FINAL_STAGE_TOP = 5`、`FINAL_CUMULATIVE_SHOT_CAPS`（Table VII 原样，已与 `hpc/top5_extension_caps.json` 逐项核对一致，测试钉住两者相等）、`FINAL_TOPUP_RNG_NAMESPACE`、`FINAL_TOPUP_PART_MAX_SHOTS`、`FINAL_TOPUP_SHOTS_PER_SHARD`、`final_cumulative_cap()`。
- `evaluation/final_stage.py`（新）：`final_stage_protocol`（冻结块）、`select_finalists`（排名与去重）、`topup_parts` / `split_parts` / `topup_shards`（补测分段）、`pool_entry`（按基相加）、`rank_pooled`（终局排名与冠军）、`execute_final_stage`（驱动：先确认、再补测、再合并）。
- `workflow/state.py:build_protocol_snapshot`：新参数 `final_stage_id` / `final_cumulative_shots`，把冻结块写进快照键 `final_stage`；strict 下拒绝 `final_cumulative_shots`。
- `workflow/graph.py`：`final_slow_node` 按快照分派——有 `final_stage` 块走 `_paper_final_slow_node`（新），没有的（旧版本冻结的运行续跑/重进终局）走 `_legacy_final_slow_node`（原代码不动）。新增 `_paper_final_entries`、`_paper_final_queue_record`、`_audit_paper_final_measurements`、`_write_final_ranking_csv`；`report_node` 的 `final_frontier` 改为合并后的排名，并写 `final_stage` / `final_stage_parts` / `champion`；`run_campaign` 默认 `final_stage_id = paper_top5_cumulative_v1`，新增 `final_cumulative_shots`。
- `cli.py`：新开关 `--final-cumulative-shots`（声明的变体，strict 拒绝；用于冒烟）。
- 产物：`final_slow/T12/`（确认段，沿用原目录名与 `final_confirmation_v1` 种子）、`final_slow/topup/T12_finalists_p<i>/`、`final_slow/topup/T12_reference_p<i>/`（补测段）、`final_slow/final_ranking.{json,csv}`（合并排名，权威结果）、`final_slow/final_shortlist.json`（schema `final_top5_composition_v1`）。
- 续跑：每一段都是普通的固定发数终局队列（`evaluate_promotion_queue(stage="final", resume_candidates=True)`），每段有独立目录和种子命名空间；某段没跑完时 strict 运行抛错、checkpoint 停在 final_slow 之前，下一次 `--resume` 用同样的队列与种子重跑，每段内已完成的候选直接复用。

**歧义怎么定的**
- "successfully evaluated"：搜索期测量状态为 ok 的非参考码（名人堂 `hof_summary` 的行）；被早停（racing）截断的候选也算——它们有有效的估计值，论文说早停只是"terminated without further sampling"。decode_infeasible / 构建失败的候选没有搜索期 LER，不能排名，也就不可能进终局；旧终局里"搜索期 decode_infeasible 的码在终局再试一次"（v3.6 B5 `retry_at_final`）在新终局里不存在。沿用旧终局的两条排除：不可测码；max_turns 续跑时被调低后、在新上限之后才首次测到的码。名人堂上限 `HOF_CAP = 12` ≥ 5，足够。
- 排名键：搜索期 p̂_LER 升序；平局按 Wilson 上界小者优先，再按 code hash（论文未写平局，取与旧 Top-5 工具相同的规则）。
- "distinct"：按 B3（L3343 起，论文行 1615–1624）的规范形身份（pynauty canonical hash，来自 structural ledger 或行自身），退化到 code hash。被跳过的孪生码记在 `skipped_duplicates`。
- "cumulative"：只累计终局的确认段与补测段，**不含**搜索期的那次（会早停的）测量。依据：Table VII 的标题是"top-five final confirmation"的上限；D3 末段说参考码"This gives 175,000 and 525,000 shots per basis"，这正是确认 + 补测的总数，若计入搜索期测量就超过了。"Existing completed measurements" = 已完成的确认段（以及续跑时已完成的补测段）。
- "further shots to reach the ... limits"：字面读法是每个前 5 名都补到上限，**不做**自适应目标（旧工具的 `50 000 * ceil(45 * 25 000 / f / 50 000)` 与"≥30 次失败就不补"均不是论文规则，已不用）。
- 某个补测段对某个码没有完成（例如 decode_infeasible）：该段不是 "completed measurement"，不并入；该码仍按已完成部分排名，行上 `cumulative_complete = false`。确认段都没完成的码不参与排名（`status = unconfirmed`）。
- 冠军 = 合并 p̂_LER 最小者（平局：上界小者，再按搜索名次）。参考码不参与排名、不能当冠军。
- Table VII 只列 9 个预算。不在表里的预算（如 README 冒烟用的 N = 72）没有补测，累计上限 = 25 000（`cumulative_cap_source = budget_not_in_table_vii`），这是"论文没写的保持现状"。
- 补测怎么切分不在论文里：超过 900 000（memory）/ 150 000（CNOT）发的补测拆成几段，每段约 300 发一个 shard、至少 128 个 shard（沿用 2026-10 补测高统计量时的规则），只影响执行与续跑粒度，不影响统计量。
- 终局队列里每个前 5 名的 program id 改成 `<搜索期 program id>_h<hash8>`（program id 每轮重新编号，两个前 5 名可能同名，会互相覆盖候选文件、破坏续跑）；原 id 保存在 `search_program_id`。
- **基线方法**（`search_baseline`：random / bo）：论文 D4/D5（L4576–4580，"using the same circuit-synthesis and circuit-evaluation modules as OmniQEC"；L4709 起 random baseline）——走同一终局规则，前 5 名取自它们自己的名人堂（旧规则是 6 个名额）。
- **Fig.4c 固定权重搜索**：D5，L4683（论文行 2019–2020）"Five leading candidates are retained per condition"——走同一规则；固定权重约束下名人堂里只有该权重的码。（→ §8：论文已改为 6 个、只做初始确认，代码已改。）
- **无慢环消融**（no_slow / no_feedback）：论文 Table VIII 与附录 E 只说比较"three leading codes"，没写终局怎么选。搜索期没有任何测量，"按搜索期 LER 排名"无从做起，所以名单沿用现状的来源（结构前沿按结构通道的槽位顺序），但数量按论文改为 5，确认与补测规则相同（`selection_source = structural_front_slot_order`）。

## 2. 参考码：同一协议、按各自预算的累计上限

（→ §8：论文已改为参考码报告值基于每基 50 000 / 100 000 发，代码已改；本节以下为 11-10 上午的原文。）

**论文原文**：D3，L4557–4560（论文行 1961–1964）："The two BB reference codes are evaluated under the same circuit protocols and cumulative shot budgets as the discovered codes at their corresponding physical-qubit budgets of N = 144 and N = 288. This gives 175,000 and 525,000 shots per basis for the two references in quantum memory, and 75,000 and 325,000 shots per basis in transversal-CNOT circuits, respectively."

**改了什么**
- `evaluation/baseline.py`：`RESIDENT_BASELINE_BUDGETS`（BB72 → 144，BB144 → 288）。
- `evaluation/final_stage.py:resident_reference`、`final_stage_protocol` 的 `reference` 子块（参考码的 program id、hash、对应预算、累计上限）。
- `workflow/graph.py:_paper_final_entries`：只要本运行有适用的参考码就把它加进终局（旧终局只在 `baseline_summary` 有 spec 时才加），25 000 发确认后补到**它自己预算**的上限，种子命名空间 `final_topup_v1/reference/p<i>`，与前 5 名的补测互不影响。
- 参考码在搜索期的那次测量照旧（第一轮随队列测一次，规则同候选，只是不和自己做 10 倍比较）。

**歧义怎么定的**
- 哪些运行有参考码：沿用现状 `select_resident_baseline`（N = 144 → BB72，N ≥ 288 → BB144，其余没有）。论文只说参考码在 144 / 288 评估，没说 320 / 360 的运行里要不要测；现状是测，保留。
- 上限取参考码自己的预算（不是运行的预算）：N = 320 / 360 的运行里 BB144 也只补到 525 000（memory）/ 325 000（CNOT），这正是论文给出的参考码总数。
- 生成器自己提出了与参考码完全相同的码（同 code hash）且它进了前 5：只测一次（行上 `also_reference = true`），按运行的上限补测；在 N = 144 / 288 两者上限相同，在 320 / 360 这一行会多于 525 000（边界情形，记在这里）。

## 3. 搜索期早停

**论文原文**
- D3，L4443–4459（论文行 1941–1946）："During discovery, circuit-level evaluation begins with 2,000 shots in each of the X and Z measurement bases, followed by additional batches of 250 shots. … After each batch, we estimate … Sampling continues until both bases have accumulated 30 block failures, subject to a maximum of 25,000 shots per basis."
- D3，L4460–4483（论文行 1947–1953，式 D2）："we construct Wilson score intervals [L_b, U_b] … using z = 1.96 … After each sampling batch, we compare the candidate's lower LER bound with the upper bounds of the current best measured code and the applicable reference code. If the candidate's lower bound exceeds both upper bounds by a factor of ten, its evaluation is terminated without further sampling."
- 正文 L1182–1245（论文行 774–786、式 (1)）。

**改了什么**
- `config.py`：`RACING_POLICY_PAPER_ID`、`paper_racing()`（试探 2 000、每批 250、上限 25 000、30 次失败、epsilon 9 即 10 倍、`baseline_arm=True`、`max_errors_in_pilot=False`、无解码代价上限）；`RacingConfig` 新字段 `strict_exceed`、`best_bound`（取旧值时不进 `to_dict()`，旧策略与旧快照的 racing 块逐字节不变）；登记进 `FORMAL_RACING_POLICIES` 与 `RACING_POLICY_RANK`（4）。
- `evaluation/sampling.py:racing_decision`：`strict_exceed` 时要求下界**严格大于**每个可用上界的 10 倍。
- `workflow/graph.py:_hof_thresholds`：`best_bound = best_estimate` 时，"当前最好的已测码"= 名人堂里点估计最小的那一行（平局取上界小者），用它**自己**的上界；参考码上界取 `baseline_summary`，没有时取生成器自己测到的同一个码的那一行（先名人堂，再 `evidence_rows.jsonl`）。新增 `_resident_reference`。
- `evaluation/pool.py`（`evaluate_queue_with_jobs`）：不在试探期内做判断的策略（论文策略、v1/v2）忽略 `OMNIQEC_RACE_PILOT_WINDOW`，整段试探一起提交——否则窗口小于试探时，判断点所需的 shard 永远不会被提交，候选会卡死。
- `cli.py`：`run-campaign` 不给 `--racing-max-errors` 时用论文策略；给 100 / 30 时是旧的 as-run 策略（打印 PROTOCOL VARIANT）。

**歧义怎么定的**
- 判断点：2 000 发试探本身算第一批（"followed by **additional** batches"），所以第一次判断在 2 000，然后每 250 一次；试探期内不判断，30 次失败的停止也从 2 000 起判（旧 `pilot10x_maxerr30_v3` 在试探期内每 250 发就判 30 次失败，与论文不符）。
- "exceeds … by a factor of ten"：严格 `>`（旧策略是 `>=`）。
- "the current best measured code"：本运行此前已测的非参考码中点估计最好的那个，取它自己的上界（旧策略取名人堂所有行上界的最小值）。阈值在每轮开始时取快照、本轮内不变：同一轮内的候选并行测量，若随完成顺序更新阈值，结果会依赖调度、不可复现；"current" 按"本轮开始时"理解。
- "the applicable reference code"：本运行的常驻参考码（见 §2：N = 144 → BB72，N ≥ 288 → BB144）在本运行里测到的那次。"both"：两者都已测时两个都要超过；本运行没有适用参考码（60、98、180、220、260）时只和最好的已测码比；还没有任何已测码（第一轮）时只和参考码比（若已测）；两者都没有时不早停。参考码自身测量时不做 10 倍比较。
- 上下界都是式 (D2) 映射后的 p_LER 尺度（K = k 或 2k，R = 12），与代码的 `per_logical_per_round_ci` 一致。

## 4. 所有评估 R = 12

**论文原文**：正文 L1152 起（论文行 745–746）"All circuit-level evaluations use R = 12 rounds of syndrome extraction"；D2，L4326、L4332（论文行 1888、1895）；式 (1)/(D1) "R = 12"（L1243、L4458）。

**核对与改动**
- 已确认 release/final 预设 `fig2_final` / `fig3_final` 是 `fixed_Tc`（`hpc/nscc_claude_grid_v36.pbs` 的 `preset_key ROUNDS_POLICY fixed_Tc`），验证器两个 final profile 也要求 `fixed_Tc`。
- 仍有 `max_Tc_d` 默认路径，已改：`cli.py` 的 `--rounds-policy` 默认改为 `fixed_Tc`（`max_Tc_d` 成为声明的变体）；`workflow/graph.py:run_campaign` 不给 verification 时用 `VerificationConfig(rounds_policy="fixed_Tc")`。终局所有段（确认、补测、参考码）都用快照里的 `rounds_policy.final`，即 12。
- 有意保留：`VerificationConfig.rounds_policy` 这个库内数据类的默认值仍是 `max_Tc_d`，`verification_from_dict` 把缺键读成 `max_Tc_d`——这是旧快照（10-15 之前的 Fig.2 运行）的含义，改了会让旧快照被读成 R = 12；S3 消融的 golden 快照也钉着它。
- 启动脚本：非预设路径下显式设置 `ROUNDS_POLICY=max_Tc_d` 时现在会显式传给 CLI（否则会被新默认值悄悄换成 `fixed_Tc`）。

## 5. 验证器与协议文档

- `hpc/validate_fig2_campaign.py`（`--profile fig2_final|fig3_final`）：
  - `_validate_final_protocol_keys`：racing 块必须逐字节等于 `paper_racing(25000)`；`final_stage` 块必须等于 `final_stage_protocol(task, N, 25000, 128)`（Table VII）。
  - `_validate_protocol`：终局名单 schema `final_top5_composition_v1`、来源为搜索期排名、5 个（不足 5 个时 = 全部可用的不同码）、按搜索期 p̂_LER 排序。
  - `_validate_turns`：逐轮重算论文的阈值（此前各轮最好码的上界；参考码测到后其上界）并与该轮 `slow_run_config.json` 记录的 `racing_thresholds` 比对。
  - `_validate_final` → `_validate_paper_final`（新）：段落与计划完全一致（确认段 25 000 发 / 128 shard / `final_confirmation_v1`；补测段的发数、shard、命名空间、目录、成员 = 确认段 ok 的成员）；每段 run config、候选 job config、噪声点、SEC 调度、行的解码器与 T；每个合并行 = 各 ok 段之和、p̂_LER 由和重算一致；排名按合并 p̂_LER，rank 1 = 冠军（报告与 `final_ranking.json` 一致）；参考码恰好一行。
  - `_validate_circuit_rounds` 也检查补测段的候选。
  - `v3_7` / `v3_8` profile 不动（它们验证的是 as-run 的 Fig.2 数据与旧终局）。
- `docs/final_protocol_20261106.md`：加修订说明，§1 早停与终局、§2/§3 终局行、§4 Top-5 工具标为 LEGACY、§5 运行方式。
- `hpc/top5_extension.py`：标为 LEGACY；遇到快照带 `final_stage` 的运行直接拒绝（新运行的终局已在 campaign 内完成）。
- `hpc/nscc_claude_grid_v36.pbs`：预设不再设 `RACING_MAX_ERRORS=30`（论文早停是 CLI 默认），预设下给了 `RACING_MAX_ERRORS` 会被拒绝。

## 6. 按要求未改（等用户决定）

噪声模型（`uniform_paper_v1`，无 idle 噪声）、快环排序指标（kd2_v1，Q = k·d²）、晋升门槛（d_search ≥ 5、结构通道 k ≥ 2、运行时 d/√n ≥ 0.25）。

**论文没写、保留的机制**：解码超时判 `decode_infeasible`（搜索期与终局的 shard 看门狗、`*_decode_shard_timeout_cap_s` 运行时上限、首个封顶超时即中止该候选）。论文 D2 只说"Shots without an accepted prediction after the CPU substages are counted as decoding failures"，没有提到看门狗；这个机制会让个别码没有搜索期 LER（因而不能进终局），或让终局某段不完整（该段不并入，见 §1）。留给论文补一句。

## 7. 与论文仍不一致或做不到的点

- 阈值每轮开始取快照（见 §3），论文 "current" 若指"随时更新"则不一致；随时更新会使结果依赖并行调度。
- N = 320 / 360 运行里仍测 BB144 参考码（现状），论文只在 144 / 288 报告参考码。
- 参考码与某个前 5 名是同一个码时只测一次、按运行上限补测（见 §2）。（→ §8.1：已改为参考码行只合并到参考总数的前缀。）
- 名人堂里同一个码被测过两次时（正常流程不会发生：已测码与其孪生码不再入慢环），取较好的那次作为搜索期估计（沿用名人堂的现状）。
- `budget_policy` 在 N ≥ 400 时 final_shots = 50 000（不在论文预算内，未动）。

## 8. 2026-11-10 论文修订：参考码发数、Fig.4 c 终局（分支 `os/ref-fig4c`）

依据：当前论文附录 "Experimental setup"（§"Candidate selection, sampling and final ranking" 与 §"Additional experiments and API accounting"）。主搜索终局那段原文未变（"the five leading distinct codes from each discovery run are selected for final confirmation … initial confirmation measurement uses 25,000 shots per basis, followed by further shots to reach the budget-dependent cumulative limits in Table VII …"），§1 的实现不动。

### 8.1 参考码：每基 50 000（BB72）/ 100 000（BB144），两个任务相同

**论文原文**："The two BB reference codes are evaluated under the same circuit protocols as the discovered codes at their corresponding physical-qubit budgets of N=144 and N=288. In both the quantum-memory and transversal-CNOT experiments, the reported LERs for the BB [[72,12,6]] and [[144,12,12]] reference codes are based on 50,000 and 100,000 shots per basis, respectively."

旧论文写的是"same circuit protocols and cumulative shot budgets"，即参考码补到各自预算的 Table VII 上限（memory 175 000 / 525 000，CNOT 75 000 / 325 000），§2 按那句实现。

**改了什么**
- `config.py`：新常量 `REFERENCE_CODE_SHOTS_PER_BASIS = {"baseline_bb_72_12_6": 50_000, "baseline_bb_144_12_12": 100_000}`（按参考码的 program id 取，与 `evaluation/baseline.py:RESIDENT_BASELINE_BUDGETS` 的键一致，测试钉住）。
- `evaluation/final_stage.py`：`resident_reference` 多返回 `shots_per_basis`；新函数 `_reference_block`（`final_stage_protocol` 的 `reference` 子块）：`cumulative_cap` = 上面的总数，`cumulative_cap_source = "paper_reference_shots"`（冒烟用的 `--final-cumulative-shots` 仍同时替换它，source `override`），新键 `shared_with_finalist = "reference_prefix_of_finalist_parts_v1"`（下面的边界规则）。参考码仍是 25 000 发初始确认 + 补测到总数，补测段命名空间 `final_topup_v1/reference/p<i>` 不变：BB72 补 25 000，BB144 补 75 000。
- 适用范围不变：凡是测常驻参考码的运行（N = 144 → BB72，N ≥ 288 → BB144，包括 N = 320 / 360、主搜索、消融、基线、Fig.4 c），两个任务都用同一个总数；参考码在搜索期的那次测量和早停比较（"the applicable reference code"）都不动。
- `workflow/graph.py:_paper_final_slow_node`：`final_ranking.json` 的 `reference_rows` 只列参考码行（`is_baseline`），带 `shared_with_finalist`、`measured_as`、`shots_per_basis`；`_audit_paper_final_measurements` 改为按每个合并行自己列出的段（`parts`）、在 `measured_as` 名下求和核对；`FINAL_RANKING_CSV_COLUMNS` 加 `shared_with_finalist`、`measured_as` 两列；`_report_markdown` 写出参考码总数。

**边界情形：生成器自己的码就是参考码（同 code hash，`also_reference`）且进了前 5 名**（`evaluation/final_stage.py:plan_final_stage`）
- 这个码只测一次，按一条段序列：25 000 发初始确认 → 参考码自己的补测段补到参考码总数（`reference` 组）→ 再补到本运行的 Table VII 上限（新组 `shared_finalist`，命名空间 `final_topup_v1/shared_finalist/p<i>`，目录 `topup/T12_shared_finalist_p<i>`）。
- 前 5 名那一行合并全部段，总数 = 运行的 Table VII 上限，参与排名、可以当冠军（`also_reference = true`）。
- 报告的参考码行（`is_baseline = true`、`program_id` = 参考码 id、`shared_with_finalist = true`、`measured_as` = 那个前 5 名的终局 program id）只合并前两部分（确认 + 参考码补测段），总数正好 50 000 / 100 000，不参与排名。
- 这样两行都是"同一组发数的前缀"，不多测一个码，参考码行在所有运行里都是同样的段和同样的命名空间。例：N = 144 memory，确认 25 000 → 参考补测 25 000 → 延伸 125 000，参考码行 50 000、前 5 名行 175 000；N = 288 CNOT，确认 25 000 → 参考补测 75 000 → 延伸 2 × 112 500，参考码行 100 000、前 5 名行 325 000。
- 前 5 名总数等于初始确认时（Fig.4 c 的终局、或不在 Table VII 里的预算）：前 5 名行只合并确认段，参考码行仍是确认 + 参考补测。前 5 名总数严格介于确认和参考总数之间的情形按现有数字不会出现（有参考码的预算点 Table VII 都大于参考总数；`--final-cumulative-shots` 同时替换两者），出现时 `plan_final_stage` 直接报错，不自行折中。
- 旧规则（§2 末条）是"只测一次、按运行上限补测、参考码行就是那一行"，N = 320 / 360 时参考码行会多于论文总数；新规则下不会。
- §1 "cumulative" 一条的依据之一是旧论文的 "This gives 175,000 and 525,000 shots per basis"，新论文已删去这句；该读法（只累计终局的确认段与补测段，不含搜索期测量）仍由 Table VII 的标题 "Cumulative shot limits per measurement basis for top-five final confirmation" 支持，未改。

### 8.2 Fig.4 c 固定权重搜索：6 个候选、只做初始确认、报告前 3

**论文原文**："For each budget, separate searches require the maximum stabilizer-check weight of a candidate code to be exactly w_max = 4, 5, 6 or 7. Each condition uses one discovery run of 30 iterations, with the same program-generation and circuit-evaluation settings as the main searches. Six candidates are evaluated in the final stage for each condition. The figure reports the mean LER and the minimum–maximum range of the three best codes under each weight constraint."

此前代码让固定权重的运行走主搜索的前 5 名 + Table VII 累计补测（§1 末的 Fig.4c 条）。

**改了什么**
- `config.py`：`FINAL_STAGE_FIXED_WEIGHT_ID = "paper_fig4c_top6_confirm_v1"`、`FINAL_STAGE_FIXED_WEIGHT_TOP = 6`、`FINAL_STAGE_FIXED_WEIGHT_REPORTED = 3`、`FINAL_STAGE_IDS`。
- `evaluation/final_stage.py:final_stage_protocol`：新参数 `exact_check_weight`；给了就返回固定权重终局块：`id = paper_fig4c_top6_confirm_v1`、`exact_check_weight = w`、`top = 6`、`reported_top = 3`、`reported_statistic = arithmetic_mean_min_max_v1`、`confirm_shots = 25 000`、`cumulative_cap = None`、`cumulative_cap_source = none_confirmation_only`、`reference` 同 8.1。`is_paper_final_stage` 认两个 id；新增 `is_fixed_weight_stage`、`finalist_total`（无上限时 = 初始确认）、`plan_final_stage`（两种终局共用的补测计划：此终局没有前 6 名的补测段）、`reported_top`（前 3 名的行与 p_LER 的算术平均、最小、最大）；`execute_final_stage` 改为按 `plan_final_stage` 执行与合并；`topup_parts` 新增 `start` 参数（延伸段从参考总数起算）。
- 选择：`workflow/state.py:build_protocol_snapshot`——有精确权重约束（`verification.exact_check_weight`）的运行由约束自动选这个终局：调用方传 `paper_top5_cumulative_v1`（`run_campaign` 的默认值）或 `paper_fig4c_top6_confirm_v1` 都得到它；没有约束却点名固定权重终局会报错。CLI（`run-campaign --exact-check-weight w`）和 PBS 预设路径（`PRESET=fig3_final EXACT_CHECK_WEIGHT=w`，它调用同一个 CLI）因此自动选上，快照键 `final_stage` 记下 id、名额 6、`cumulative_cap: null`。`cli.py` 的帮助和 PROTOCOL VARIANT 提示、PBS 头注释同步。
- 选码规则与主搜索相同（`workflow/graph.py:_paper_final_entries` 的通用分支，名额从块里读）：名人堂按搜索期 p̂_LER 排序（平局按上界、再按 code hash），按规范形身份去重、退化到 code hash，取前 6 个。每个只测一次 25 000 发/基（命名空间 `final_confirmation_v1`、128 shard，同主搜索的确认段），不补测；按这次测量排名，rank 1 为冠军。
- 输出：`final_slow/final_ranking.json` 与 `final_slow_summary.json` 多一个 `reported_top`（前 3 名的 program id、hash、发数、失败数、p_LER 与区间，以及三者 p_LER 的算术平均 / 最小 / 最大；不足 3 个时 `complete = false`），`campaign_report.json` 的 `final_reported_top` 同值。Fig.4 c 的柱子和误差条直接取这里。
- 参考码：N = 144 的条件照 8.1 测 BB72（25 000 + 补测到 50 000），不进排名、不进前 3；N = 98 / 180 没有参考码。
- 验证器 `hpc/validate_fig2_campaign.py`：新参数 `--exact-check-weight w`（只配 `--profile fig3_final`，3 ≤ w ≤ 8）。给了它：快照必须带 `exact_check_weight = w`（顶层与 verification 两处），慢环门槛与快环上限都是 w，`final_stage` 必须逐项等于 `final_stage_protocol(…, exact_check_weight=w)`，名单 6 个（`_validate_protocol`），段落、成员、每行合并和 `reported_top` 由 `_validate_paper_final` 按 `plan_final_stage` 重算核对。不给它时 final profile 仍拒绝带 `exact_check_weight` 的运行（原行为）。启动器里 `EXACT_CHECK_WEIGHT` 仍要求 `VALIDATE_FORMAL=0`（没改），跑完后手动用上面的参数验证。
- 主搜索、消融、BO/Random 基线：终局仍是 `paper_top5_cumulative_v1`，名单规则、Table VII 上限都没变（测试钉住）；只有其中参考码的总数按 8.1 变了。

**歧义与未定的点**
- 论文只说"Six candidates are evaluated in the final stage"，没写这 6 个怎么选、测多少发。按"with the same … circuit-evaluation settings as the main searches"取：与主搜索同一选码规则（搜索期 p̂_LER 前 6 个不同的码）、同一初始确认（25 000 发/基），不补测（"evaluated"，且 Table VII 标题是 "top-five final confirmation"）。当初实际跑的 Fig.4 c 用的是旧终局（2 个结构端点 + 4 个实测最好，每个 25 000 发），名单来源与这里不同。
- "the three best codes"：按终局那次测量（不是搜索期估计）排的前 3。
- "the mean LER"：取算术平均（与 S3 消融 Top-3 的用户规则相同，`hpc/s3_ablation_top3.py`）；若图上用的是对数尺度上的平均（几何平均），需要改 `evaluation/final_stage.py:reported_top` 的统计量和 `REPORTED_TOP_STATISTIC`。
- 论文没说 Fig.4 c 的条件里报不报参考码；这里在 N = 144 照测（与主搜索相同的 50 000 发），只作对照。
- 参考码"based on 50,000 and 100,000 shots per basis"：按任务说明实现为 25 000 发初始确认 + 补测到总数（合并）。`docs/fig2_formal_protocol_final_20261030.md` §9 和 `docs/fig3_formal_protocol_final_20261030.md` §9 记的当初做法是参考码只报补测那一段（50 000 / 100 000 发新种子，不与 25 000 发的终局合并）；总发数相同，但不是同一组发。


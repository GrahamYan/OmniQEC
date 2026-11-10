# Fig.2 实验协议：每个数字的来源（写论文用）

整理日期 2026-10-15。适用对象：Fig.2 的 27 个正式运行（协议 `telescoping_structural_v3_7`，源码标签 `exp/fig2-v37` = 7e0bb44），
归档在 `runs_export/fig2_v37_final_20261014/`（`runs/` 27 个运行目录、`tables/` 画图表、`provenance/` 验证与清单、`README.md`）。
Fig.3 用的是 v3.7.1，不在本文范围。

每一项给出：数值、怎么定的（日期与依据）、文献支撑（如有）、在数据里的记录位置。"用户定案"指课题负责人当天的决定，
依据写在旁边；"代码常量"指 `omniqec_agentic/` 下的冻结常量，验证器（`provenance/fig2_validate_706adc39.py`）逐运行核对过。

---

## 1. 预算点、臂与模型

| 项 | 值 | 来源 / 依据 | 记录位置 |
|---|---|---|---|
| 预算点 N | 60, 98, 144, 180, 220, 260, 288, 320, 360 | 用户定案 2026-09-10（三档：60/98/144 → 180/220/260 → 288/320/360，逐档 gate 后再上下一档）。144 与 288 取自 BB72、BB144 的总比特数，98 取自 Quantinuum Helios 的物理比特数 | `protocol_snapshot.json: n_budget`；`provenance/run_table.md` |
| 预算含义 | N = 数据比特 + 每个校验行一个测量辅助比特（N_data + N_checks ≤ N） | 代码：`config.py` `budget_kind: "total"`；LLM 提示里同样写明 | `campaign_spec.json: n_budget, budget_kind` |
| 臂 | Claude = `claude-opus-5`；DeepSeek = `deepseek/deepseek-v4-pro`；GPT = `gpt-5.6-sol` | 用户定案 2026-09-11（Claude 臂是 opus-5 不是 Fable）。路由（OpenRouter / 共享代理）不是协议的一部分 | `protocol_snapshot.json: required_model, required_provider` |
| 每点每臂 1 个运行 | 27 = 9 × 3 | 同上 | `provenance/run_table.md`（含账号与完成日期） |
| 轮数 | 30 轮 | 用户定案 2026-09-05（60 → 30，同时把每轮构筑量提回 60 轮水平，见 §4） | `protocol_snapshot.json: max_turns` |

## 2. 噪声、线路与解码器

| 项 | 值 | 来源 / 依据 | 记录位置 |
|---|---|---|---|
| 噪声模型 | `uniform_paper_v1`，单一物理错误率 p | OmniQEC 论文 SI C "uniform v1"（代码 `verification/noise.py: UNIFORM_PAPER_SOURCE`） | `protocol_snapshot.json: verification.noise` |
| p | 0.5 % | 用户定案 2026-09-10"p=0.5% 不变"（沿用 v3.4 起的锚点） | `verification.noise.values = [0.005]`，`anchor_p` |
| 记忆实验 | X 基与 Z 基各一个记忆实验，每轮都测两基、不交织（X 半程后 Z 半程） | SEC 策略 `twobasis_nonint_v1`，v3.5 定案 2026-09-02 | `verification.sec_policy` |
| 综合征提取线路 (SEC) | 每基 16 个候选层序，ISD 认证 400 次迭代（× 扩展核维数 × 4 取大）、3 个种子取最小、前 4 名用 2000 次迭代复核 | 代码常量 `SecPolicyConfig`（v3.6，docs/06 §3 M1） | 同上；每个记忆实验的 schedule blob |
| 记忆轮数 T | T = max(12, d_search)，搜索与终局同规则；T 轮带噪综合征提取后直接测量数据比特（测量前有概率 p 的翻转），由数据读出重建最后一层探测器；**没有额外的无噪综合征轮**（10-30 更正：此处原写'接一轮无噪最终读出'，与线路不符，核对见 `fig2_formal_protocol_final_20261030.md` 附注） | 两部分：(a) **T = d 是文献惯例**（Bravyi 2024 原话 "Following common practice, we choose Nc = d for a distance-d code"；Relay-BP、Maan 2025 同样 d 轮；Stephens 2014：d 轮才能让辅助比特错误获得与数据比特错误同等的保护）；(b) **下限 12 是本协议的公平性规则**（代码 `verification/circuit.py: DEFAULT_SEARCH_ROUNDS` 注释，2026-08-16）：单次 T 轮实验推出的每轮错误率带 O(1/T) 的时间边界瞬态（首末轮无噪准备/读出），T 小的码被系统性高估，所以所有候选与基线至少共用同一深度，12 = 参考码 BB144 的 d；d > 12 的码按 T = d 测，时间向错误不被截断。2026-09-10 用户定案把 "固定 12" 改成 max(12, d)。文献对边界瞬态的描述见 §8 | `rounds_policy.search/final = "max_Tc_d"`，`search_rounds/final_rounds = 12`（下限）；每条测量行的 `rounds` |
| 解码器 | telescoping 解码器 CPU 配置 `telescoping_cpu_v1`，包 `telescoping-decoder@c97099b759dc`，S3 用 GARI 系统；S1/S2/S4 关 | v3.6 定案 2026-09-03：正式解码器 = 组内 Mitten 论文开源的 telescoping 源码；GARI 见 Maan 等 2025（§8） | `verification.decoder_build.telescoping`；`resolved_versions.json` |
| 不收敛策略 | `nc_policy = "fail"`（BP 不收敛的 shot 计为失败） | 包默认 | 同上 |

## 3. 指标与统计口径

| 项 | 值 | 来源 / 依据 | 记录位置 |
|---|---|---|---|
| 报告指标 plpr | 每逻辑比特每轮失败率：block = 1 − (1 − p_X)(1 − p_Z)；plpr = 1 − (1 − block)^(1/(k·T)) | OmniQEC 论文附录 A4（两基合并）、A5（开根到每逻辑每轮）；代码 `verification/ler.py: combine_basis_ler_results` | 每条测量行 `per_logical_per_round`；README 第 6 条 |
| 置信区间 | 每基 Wilson 区间，z = 1.96（95 %），按 A4/A5 映射到 plpr 尺度 | 代码常量 `ranking.DEFAULT_CI_Z`；Wilson 区间是二项比例的标准区间 | `per_logical_per_round_ci`、`basis_ler_ci` |
| 搜索质量分 Q | Q = k · d_search² | 用户定案 2026-09-10（`QUALITY_METRIC_ID = "kd2_v1"`）：原来的 k·d²/n 与尺度无关，半预算小码与满预算码同分，系统性偏向小码（审计见记忆 `fig2-small-code-dominance-audit`） | `protocol_snapshot.json: quality_metric` |
| 代理距离 d_search | `css_permutation_systematic_v2_independent_xz`，种子 0–3，每种子 120 次迭代，X/Z 各自独立 | 代码常量 `StructuralSearchConfig.proxy_*`（每个 family 同样的种子与算力） | `structural_ledger.jsonl` 每行的 proxy 记录 |
| 精确距离 | n ≤ 100 的码用 MILP 求码距（上限 8），否则只有 d_search | 代码常量 `ExactDistanceConfig` | 终局表 `exact_code_distance`, `exact_code_distance_status` |

## 4. 搜索阶段（每轮）

| 项 | 值 | 来源 / 依据 | 记录位置 |
|---|---|---|---|
| LLM 提案 | 每轮 24 个程序 × 4 代 = 96 个程序 | 用户定案 2026-09-10 傍晚：30 轮下把构筑总量提回 60 轮水平（原 24 × 2） | `proposal_count`, `generations_per_turn` |
| 随机种子 | 每个程序 40 个 rng 种子 → 每轮最多 3840 次结构评估 | 同上（依据：慢测码来自种子 > 0 的比例 DeepSeek 85 %、Claude 6–7 %、GPT 0 %，确定性臂每轮只有 24 个不同码） | `rng_seeds_per_program` |
| 结构前沿 | 三轴非支配 (k, d_search, −max_check_weight)，精英池 24 | 用户定案 2026-09-12（v3.7）：否决权重上限与权重分箱，改为把权重作为前沿第三轴 | `structural_retention.structural_axes, elites_limit` |
| 晋级门槛 | d_search ≥ 5；k ≥ 2 | 用户定案 2026-09-10 傍晚（依据 v3.6 三臂 CSV：7 个点最好码全是 d_search ≥ 5，d ≤ 4 占 34–60 % 慢测且从未进 1.5 倍内）；k ≥ 2 是 v3.7 定案。已知小漏洞：k ≥ 2 只在结构通道生效，运行时通道约 1 % 名额给了 k = 1 的码，三臂同规则 | `rounds_policy.min_promotion_distance`, `structural_retention.min_promotion_k` |
| 慢环名额 | 每轮 6 = 2 个结构端点（最大 k、最大 d_search）+ 4 个运行时候选（按 `expected_plpr_v1` 预期 plpr 排） | 用户定案 2026-09-12（v3.7；之前 4S+2R 让 4/6 名额落到 w = 8/9 的码上） | 每轮 `slow_queue_composition.json`；`structural_retention.structural_slow_slots / runtime_slow_slots` |
| 父代 | 16 = 6 结构 + 6 运行时（每权重类 ≤ 2）+ 4 多样性 | 同上 | `structural_retention.parent_cap, *_parents` |
| 校验权重 | 无上限（`max_check_weight = null`）；**例外：≥288 的 Claude/DeepSeek 臂中途加 "w ≥ 9 不进慢环"**（`promotion_max_check_weight = 8`） | v3.7 定案去掉上限；gate 是用户定案 2026-09-29：763 个二三档慢环候选中，好码（plpr < 2e-4，413 个）最大 w = 8，w ≥ 9 的 141 个最低 plpr 2.9e-4、却占慢环 CPU 42 % | `verification.max_check_weight`；`runtime_overrides` 中 `promotion_max_check_weight`；`slow_queue_composition.json: weight_gate` |
| 搜索发数 | 目标 25k/基（总 5 万） | 用户定案 2026-09-10："5 万总 shot 已经够准"（原 4 万） | `budget_policy.search_shots`；每行 `shots` |
| 分片 | 16 片/基 | 代码常量 | `shot_shards` |
| 早停 pilot | 先测 2000 发/基（每片 250 发） | 用户定案 2026-09-11（`pilot10x_v1`） | `racing.shots_pilot, batch` |
| 早停 10 倍规则 | 候选 plpr 区间下界 ≥ (1 + 9) × 名人堂最好码区间上界 → 淘汰（`stopped_by = racing`） | 同上；`RACING_EPSILON = 9.0`；基线不作参照（`baseline_arm = false`）。第 0 轮名人堂为空时不淘汰任何码 | `racing.epsilon`；每行 `stopped_by` |
| 早停 100 次失败 | 每基累计 100 次失败即停（`max_errors = 100`，`pilot10x_maxerr100_v2`，v3 起 pilot 内也判） | 用户定案 2026-09-11 20:00："每个保证有几十个 error 就够了，终局测 25k/25k"。相对标准误差 ≈ 1/√100 = 10 %，与文献一致（§8） | `racing.max_errors, policy_id` |
| 早停 30 次失败 | **仅 ≥288 的 Claude/DeepSeek 臂**：`max_errors = 30`（`pilot10x_maxerr30_v3`） | 用户定案 2026-09-29：三档一轮 6–29 h，与权重 gate 一起上；30 次对应相对误差 18 %。当时分析：省时间的主要是 gate，30 本身省得少 | `runtime_overrides`；每轮 `slow_run_config.json: racing.policy_id` |
| 早停切入轮 | Claude b288 第 14 轮、b320 第 9、b360 第 9；DeepSeek b288 第 9、b320 第 17、b360 第 12（gate 与 30 同一轮）；GPT 三档全程 100、无 gate | 由各链当时所在轮决定 | `runtime_overrides[].applies_from_turn` |
| v4 代价上限 | **10-30 更正：按每轮实际配置统计，14 个运行共 69 轮在 v4 下（GPT b180 第 14–29 轮、b220 第 13–27 轮、b260 第 11–25 轮、Claude b320 第 2–8 轮，其余 10 个运行各 1–3 轮），共 19 个候选被它判为 decode_infeasible；下面原文的'一夜'只是启用到撤回命令的时间，各链实际切回的轮次不同。** 曾在 2026-09-14 夜到 09-15 早启用"0.10 CPU-s/shot/qubit 超限判 decode_infeasible"（`pilot10x_maxerr100_cost_v4`），次日撤回 | 热身时间被计入、三档正经码本身 14 s/shot 贴线，误伤 16 个候选 → 撤回；GPT b180 第 14 轮后到结束一直在 v4 下 | `runtime_overrides`；`decode_infeasible_hashes.json` |
| 搜索分片看门狗上限 | 28 800 s（8 h） | 2026-09-25（b360 一个候选卡 12 h）；只有 Claude/DeepSeek 运行有，GPT 没有 | `runtime_overrides: search_decode_shard_timeout_cap_s` |

搜索慢测实际怎么结束（归档账本 `tables/fig2_search_ledger_all_turns.csv`，4361 条，不含基线；"≤2000/基"= pilot 淘汰，"25k/基"= 跑满，其余 = 失败数阈值停）：

| 档 | 臂 | 跑满 25k/基 | 阈值停 | pilot 淘汰 |
|---|---|---|---|---|
| 60–144 | Claude / DeepSeek / GPT | 12 % / 9 % / 3 % | 56 % / 32 % / 24 % | 32 % / 60 % / 73 % |
| 180–260 | Claude / DeepSeek / GPT | 73 % / 44 % / 55 % | 12 % / 25 % / 21 % | 15 % / 32 % / 25 % |
| ≥288 | Claude / DeepSeek / GPT | 78 % / 66 % / 47 % | 9 % / 14 % / 28 % | 13 % / 20 % / 25 % |

大码失败率低，多数跑满 25k 也攒不到阈值，所以 100 与 30 的差别只落在中等码上。

## 5. 终局（每个运行一次）

| 项 | 值 | 来源 / 依据 | 记录位置 |
|---|---|---|---|
| 决赛名单 | 2 个结构端点 + 4 个实测 plpr 最好 + 常驻基线 | v3.7 定案 2026-09-12 | `final_slow/final_shortlist.json`；`structural_retention.final_structural / final_runtime` |
| 基线 | N = 144 带 BB72 [[72,12,6]]；N ≥ 288 带 BB144 [[144,12,12]]；144 与 288 之间无基线 | 代码 `evaluation/baseline.py: select_resident_baseline`（BB72 占 144 总比特、BB144 占 288） | 终局表 `is_baseline`；`tables/fig2_baseline_per_run.csv` |
| 终局发数 | 每码固定 25k/基（总 5 万），不早停 | 用户定案 2026-09-10（原方案终局 10 倍，改为与搜索相同） | `budget_policy.final_shots`；`final_slow/T<r>/slow_run_config.json: racing = null` |
| 分片 | 128 片/基 | 代码常量 | `final_shot_shards` |
| 种子 | 终局独立种子命名空间 `final_confirmation_v1` | 代码常量 | `rounds_policy.final_rng_namespace` |
| 冠军 | 每点每臂 top-1：终局非基线码中 plpr 最低者，并列取区间上界更小者 | `provenance/build_tables.py` | `tables/fig2_best_per_point_arm.csv` |
| 终局分片看门狗 | Claude/DeepSeek 带 14 400 s 上限（2026-09-24 起）；≥288 六个运行 10-14 补跑时改为 2 592 000 s（等于无上限）；GPT 终局 09-17 前跑完，从未有上限 | 上限 + 300 s 分块地板让一发不收敛就整片作废（24 804/25 000 发）→ 候选被判 decode_infeasible，故补跑。补跑只重测被误判的运行时码，结构端点保留原记录 | `runtime_overrides: final_decode_shard_timeout_cap_s`；每个补跑运行内 `_final_attempt1_capped_20261014/`；README 第 1 条 |
| 25k 下的终局失败数 | b60：256–870/基；b98：79–105；b144：37–103；b180：9–20；b220 以上 0–9 | 实测 | `tables/fig2_final_stage_all.csv: failures_x, failures_z` |

## 6. 补测（协议外补充，只针对已选定的冠军）

目标：每基至少 30 次失败（用户定案 2026-10-14："这些大码有 30 个 error 就行"）。30 次 → 相对标准误差 18 %，95 % 区间约 ×/÷1.4。
路径：`hpc/fig2_extended_stats.py`，用源运行的 `protocol_snapshot.json` + `campaign_spec.json` 重建终局校验配置，只改发数，固定发数不早停，
独立种子命名空间，无分片上限；结果与正式 25k 计数相加（两段都是同设置下的固定发数独立样本）。冠军不重选。

| 轮 | 规则 | 发数/基 | 作业 |
|---|---|---|---|
| 1（10-14） | 按码定：r = (X 失败 + Z 失败)/50 000（正式 25k 的两基合计率），发数 = 45/r 向上取整到 5 万倍数，封顶 50 万；0 失败的码（b360 Claude/DeepSeek）直接 50 万 | 150k–500k | 25805235–25805252（17 个：b220–b360 15 冠军 + BB72 50k + BB144 100k） |
| 2（10-14） | 第 1 轮后每基仍 < 30 的三个码，用第 1 轮实测率按同一公式补差额 | b320 Claude +350k、b360 Claude +1.2M、b360 DeepSeek +500k | 25807974–25807976，命名空间 `extended_stats_v1_r2` |
| 3（10-15） | **按点统一**（用户 10-15 意见："每个点统一补测多少个 shot"）：每点三臂的补测发数取该点已测的最大值，不足的臂补差额；N = 180 新增（正式 25k 只有 9–20 次失败），按最弱一基（DeepSeek X 9/25k）凑 45 次 → 125k，取 150k | b180 150k；b220 250k；b260 500k；b288 500k；b320 850k；b360 1.7M | 25825264–25825275（12 个），命名空间 `extended_stats_v1_r3`（b180 三个新码用 `extended_stats_v1`） |

三轮已全部完成（2026-10-15 20:41）。补后每点三臂的补测发数一致，每个冠军合并后每基 ≥ 34 次失败，区间宽 1.4–1.9 倍；画图 N ≥ 180 用 `tables/fig2_extended_stats.csv` 的 `pooled_*` 列（正式 25k + 补测），N ≤ 144 用正式终局值。
参考线 BB72 = 50k/基、BB144 = 100k/基，只用补测段。统计学依据与文献见 §8。计划文件 `hpc/fig2_extended_stats_plan_20261014.json`
（`rounds` 键记录三轮规则），作业日志 NSCC `~/fig2_extended_stats_jobs_20261014.txt`，归档 `provenance/extended_stats_jobs_20261014.txt`。

## 7. 论文里要交代的不对称

1. ≥288 的 Claude/DeepSeek 臂中途启用 w ≤ 8 慢环 gate 与 30 次失败早停；GPT 臂三档全程 100 次、无 gate。
2. 搜索/终局分片看门狗上限只有 Claude/DeepSeek 运行带过；GPT 没有。终局上限导致的误判已通过 10-14 补跑消除，剩余 24 个 decode_infeasible 全是结构端点（w 8–18，搜索期 plpr 1e-3–1e-1）。
3. v4 代价上限的误伤（10-30 按每轮配置重数：14 个运行共 69 轮在 v4 下，19 个候选被判 decode_infeasible；原记 16 个）分布在 GPT b180/b220/b260/b320、DeepSeek b220/b260/b360、Claude b320。
4. 搜索期 plpr 精度不一（阈值停 vs 跑满），账本里的搜索值只用于排名，不作报告值。
5. 冠军数值带赢家回归：25k 下按最小值选出的码，其正式值偏低，补测后普遍上移（b320 DeepSeek 1.33e-6 → 3.87e-6，b288 GPT 3.33e-6 → 6.03e-6）。
6. b320 Claude 终局前三名并列（各 1/1 次失败，1.43e-6），top-1 由并列规则决定。

## 8. 文献支撑

**100 次失败（搜索早停阈值）**
- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, *High-threshold and low-overhead fault-tolerant quantum memory*, Nature 627, 778 (2024), arXiv:2308.07915, §8："All data points except for those with the smallest error rate accumulated at least 100 logical errors to estimate the logical error rate pL with the error bars ≈ pL/10."（BB144 参考码的来源论文）
- Müller et al., *Improved belief propagation is sufficient for real-time decoding of quantum memory* (Relay-BP), arXiv:2506.01779, §V："Each tile uses the smaller of 500,000 samples or a number achieving ≤10% relative accuracy."
- Beverland, Carroll, Cross, Yoder, *Fail fast: techniques to probe rare events in quantum error correction*, arXiv:2511.15177, Fig. 7："Monte Carlo samples include 100 logical fails per point (and 1000 fails at p0) and so have negligible error."
- 工具：Stim 的采样器 sinter 以 `max_errors` 为标准停止条件（"Stops the sampling process after this many errors have been seen"，https://github.com/quantumlib/Stim/blob/main/doc/sinter_api.md）；Google Tesseract 解码器仓库示例 `--max_errors 100`（https://github.com/quantumlib/tesseract-decoder）。

**30 次失败（三档放宽）**
- Maan, Garcia-Herrero, Paler, Savin, *Decoding Correlated Errors in Quantum LDPC Codes*, arXiv:2510.14060（Nat. Commun. 2026）："Error bars in all plots indicate 99% confidence intervals, calculated from at least 100 decoding failures, except for the Relay-BP and Tesseract decoders, for which the number of decoding failures is about 40 at low physical error rates."（GARI 即本协议解码器 S3 所用的图变换）
- Nagayama, Fowler, Horsman, Devitt, Van Meter, *Surface code error correction on a defective lattice*, New J. Phys. 19, 023050 (2017)：每个数据点 50–1500 个逻辑错误。
- 更保守的老标准：Fowler et al., *Towards practical classical processing for the surface code*, arXiv:1110.5133：每点 10 000 个逻辑错误。

**按失败数停止的统计学（逆二项抽样）**
- Haldane, *On a method of estimating frequencies*, Biometrika 33, 222–225 (1945)：固定成功数停止的抽样，无偏估计 (m−1)/(n−1)。
- Mendo & Hernando, *Estimation of a probability with optimum guaranteed confidence in inverse binomial sampling*, arXiv:0809.2402（Bernoulli）。
- 计数的相对标准误差 ≈ 1/√F：100 → 10 %，40 → 16 %，30 → 18 %。

**记忆轮数 T = max(12, d) 与每轮归一**
- T = d 的惯例：Bravyi et al. 2024（arXiv:2308.07915 §3）："Let PL(Nc) be the logical error probability after performing Nc syndrome cycles. Define the logical error rate as pL = 1 − (1 − PL(Nc))^(1/Nc) ≈ PL(Nc)/Nc ... Following common practice, we choose Nc = d for a distance-d code."；同文 §7："adding one noiseless cycle at the end of the syndrome measurement circuit, which is a common practice in numerical simulations of error correction."
- Müller et al. 2025（Relay-BP，arXiv:2506.01779 §IV）："each simulation uses d noisy QEC cycles with strength-p circuit noise followed by one perfect cycle"；gross 码即 12 个周期。
- Maan et al. 2025（arXiv:2510.14060 §III）："the circuit obtained by repeating the syndrome measurement d times, followed by a final measurement of the data qubits"；每轮归一 Eq. (7)。
- Stephens, *Fault-tolerant thresholds for quantum error correction with the surface code*, PRA 89, 022321 (2014), arXiv:1311.5003："We define the logical error rate to be the error rate per d rounds of measurement, following Raussendorf et al. This definition reflects the fact that, for a roughly isotropic noise model, d rounds are required to achieve the same protection against errors affecting ancillary qubits as against errors affecting data qubits."
- 下限 12 的依据（时间边界瞬态）：Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature 614, 676 (2023), arXiv:2207.06431："We fit the logical fidelity F = 1 − 2pL to an exponential decay, starting at t = 3 to avoid time-boundary effects that are advantageous to the distance-5 code."；Low et al., *A Denser Planar Surface Code*, arXiv:2605.30455（Google Quantum AI）：无噪初态制备与末态读出的假设 "leads to a mild underestimate compared to including noisy state preparation and measurement or running the memory experiment run longer for, say, 20d rounds"。本协议用固定下限（所有候选与参考码同深度）而不是逐码拟合，是因为一轮里要测 6 个候选且每码只测一个 T；它是比较内部一致的保守做法，不是文献数字。若审稿要求，可用 BB72 在 T = 6/12/24/48 下做一次标定，给出 T = d 相对大 T 的每轮率偏差（未做）。
- 每逻辑比特每轮归一：Bravyi 2024 的 pL 公式（每周期）再按 k 开根，即 1 − (1 − P)^(1/(k·T))；同一做法见 arXiv:2606.27119（"converts the total block-level logical error probability into the effective logical error probability of a single logical qubit ... to enable fair comparison between code families with different logical encoding rates"）与 Low et al. 2026（"logical error rate per round per logical qubit"）。
- 论文建议写法：每个码的记忆实验跑 T = max(12, d) 轮带噪综合征提取，随后直接读出数据比特（读出带测量翻转噪声，没有无噪轮；10-30 更正）；T = d 遵循 Nc = d 的惯例 [Bravyi 2024]，T ≥ 12 的共同下限使小距离码不因时间边界效应 [Google 2023] 获得优势，12 等于参考码 BB144 的距离；报告值为每逻辑比特每轮失败率 1 − (1 − P)^(1/(kT))，两基按 A4 合并。
- 两点要如实交代：下限 12 是本协议自定的公平性规则，不是文献数字；d > 12 的码只测 T = d，没有测更长的 T。

**pilot 10 倍淘汰**
- Maron & Moore, *Hoeffding Races: Accelerating Model Selection Search for Classification and Function Approximation*, NIPS 1993：用置信区间在候选比较中提前淘汰明显落后者。

**补测与合并**：补测每轮发数在测前定死、不看中途失败数停，所以每段是普通二项样本，计数相加后用 Wilson 区间即可；按点统一的发数由该点失败率最低的码决定。

## 9. 数据出处索引

| 要查什么 | 看哪里 |
|---|---|
| 一个运行的全部冻结协议 | `runs/<run>/protocol_snapshot.json`（§1–§5 的键名都在里面） |
| 中途运行时变更（早停策略、gate、看门狗上限）及切入轮 | `protocol_snapshot.json: runtime_overrides[]`（`turn`, `applies_from_turn`, `from`, `to`） |
| 某一轮实际用的策略 | `runs/<run>/turns/turn_XXXX/cycle/slow/slow_run_config.json: racing` |
| 某一轮 6 个名额怎么分的 | `turns/turn_XXXX/cycle/slow/slow_queue_composition.json` |
| 终局配置与结果 | `runs/<run>/final_slow/T<r>/{slow_run_config.json, rankings.json, candidates/}` |
| 每条搜索慢测（发数、失败数、停止原因可由发数推断） | `tables/fig2_search_ledger_all_turns.csv` |
| 终局全部候选 | `tables/fig2_final_stage_all.csv` |
| 每点每臂冠军（N ≤ 144 画图值） | `tables/fig2_best_per_point_arm.csv` |
| N ≥ 180 画图值（合并后） | `tables/fig2_extended_stats.csv: pooled_*` |
| 补测原始测量 | `extended_stats/<key>/{extended_stats_config.json, result.json, T<r>/}` |
| 27 个运行的账号、模型、完成日期、重验结果 | `provenance/run_table.md`, `provenance/revalidation_*.txt` |
| 验证器与建表脚本副本 | `provenance/fig2_validate_706adc39.py`, `provenance/build_tables.py` |

## 10. 决策时间线（只列影响数字的）

| 日期 | 决定 |
|---|---|
| 2026-09-02 | SEC 策略 twobasis_nonint_v1（两基、不交织） |
| 2026-09-03 | 解码器换成 telescoping CPU 配置（v3.6） |
| 2026-09-05 | 60 轮 → 30 轮 |
| 2026-09-10 | Q = k·d²；搜索/终局各 25k/基；T = max(12, d)；p = 0.5 % 不变；门槛 d ≥ 5；24 × 4 代 × 40 种子 |
| 2026-09-11 | 慢环早停：pilot 2000/基 + 10 倍淘汰；晚：每基 100 次失败即停 |
| 2026-09-12 | v3.7：三轴前沿、2S + 4R、16 父代、无权重上限、k ≥ 2 |
| 2026-09-14/15 | v4 代价上限启用后撤回（各链实际切回的轮次不同，共 69 轮受影响，见 §4 更正） |
| 2026-09-24/25 | 终局 / 搜索分片看门狗上限 14 400 s / 28 800 s（Claude/DeepSeek） |
| 2026-09-29 | ≥288 的 Claude/DeepSeek：w ≤ 8 gate + 30 次失败早停 |
| 2026-10-14 | 六个 ≥288 Claude/DeepSeek 终局去上限补跑；补测第 1、2 轮（每基 ≥ 30 次失败） |
| 2026-10-15 | 补测第 3 轮：按点统一发数，N = 180 加入 |

## 11. 三档复跑（2026-10-16，进行中）

用户看过合并后的图后决定对线末端明显低于 60–260 拟合线的点再跑一次搜索。用 60–260 六个点按 N 线性拟合 −log10(plpr)，288/320/360 的残差（正 = 好于拟合线）：Claude −0.12/−0.19/−0.10，DeepSeek −0.12/−0.54/−0.21，GPT −0.37/−0.68/−1.02。
据此复跑 DeepSeek b320 与 GPT b288/b320/b360 四个点（DeepSeek b288 与 Claude 偏离相同，b360 偏离在置信区间内，不跑）。

| 项 | 值 |
|---|---|
| 协议 | 与正式运行相同（telescoping_structural_v3_7），但 w ≤ 8 慢环 gate 与 30 次失败早停从第 0 轮起生效（正式三档是中途切入），搜索分片上限 8 h，终局上限 30 天（等于无） |
| 代码树 / 标签 | 集群上的部署树（= exp/fig2-v37 + 88e0cc7 的 pool.py），RUN_TAG=v37r2，目录 `fig2_<arm>_b<N>_v37r2` |
| 验证 | launcher 的正式验证只接受 RUN_TAG=v37，故 VALIDATE_FORMAL=0 启动，跑完手动运行同一验证器 |
| 作业 | 127 核提交、走 NSCC q3（q4 的每用户 25 个运行名额当天被 Fig.3 CNOT 批次占满；launcher 自 83ec4fe 起接受 127 核）：DeepSeek b320 25840777 起 3 节；GPT b288 25840780 起 3 节；GPT b320 25840783 起 3 节；GPT b360 25840786 起 3 节（10-16 17:10）。此前两批提交（25839330…、25840592…）分别因账号排队限流和 launcher 的 128/64 核检查未能启动，已删除，未产生任何运行目录 |

**论文里必须写明**：这四个点是两次独立搜索运行中的较好者，Claude 的点和其余点是单次运行；两次运行都要放进 SI。复跑是一次新的随机抽样，结果不保证更好；若更差，按"取较好"规则仍显示第一次的值。

**结果（10-17 17:40 更新）**

| 点 | 第一次（正式）pooled | 复跑冠军 | 复跑 pooled | 结论 |
|---|---|---|---|---|
| GPT b288 | [[144,4,12]] lp_abelian 8fa59d8e：6.03e-6 [4.84e-6, 7.53e-6] | g0001_p0007 [[144,4,12]] lp_abelian w6 01997335（另一个码）：终局 25k/基 3+2 次 4.17e-6；补测 500k/基 38+48 次 3.58e-6 | **3.61e-6 [2.70e-6, 4.82e-6]**（525k/基，41+50 次） | 取复跑值 3.61e-6；线值 2.58e-6 的 1.4 倍，与 Claude 3.07e-6 / DeepSeek 3.45e-6 同档 |
| GPT b320 | 4.86e-6 | 复跑在第 22 轮按用户要求停止，无更好码 | — | 保留第一次；改跑 v3.8（§12） |
| GPT b360 | 3.37e-6 | 复跑在第 21 轮停止，无更好码 | — | 保留第一次；改跑 v3.8（§12） |
| DeepSeek b320 | 4.10e-6 | g0003_p0015_s38 [[154,6,14]] lp_abelian w6 7efc7125：搜索第 15 轮 2/50k；终局 25k/基 1+2 次 1.43e-6（5 个 runtime 终选里最好）；补测 850k/基 55+60 次 1.61e-6 | **1.61e-6 [1.24e-6, 2.07e-6]**（875k/基，56+62 次） | 取复跑值 1.61e-6；线值 1.19e-6 的 1.35 倍，与 Claude 1.59e-6 同档。补测在 2 个结构端点终选（检查权重约 16、从未进慢环）仍在解码时先行提交，它们不可能排到它前面；10-19 用户决定在 5/7 时停止终局（不再测这两个端点），因此这次运行没有 campaign_report、不能跑验证器 |

GPT b288 复跑验证：`validate_fig2_campaign.py --run-dir runs/fig2_gptsub_b288_v37r2 --turns 30 --p 0.005 --provider openai --model gpt-5.6-sol --decoders telescoping_cpu` 通过（30 轮、终局 7 行、25k/25k、128 终局分片）。补测走正式补测工具（`hpc/fig2_extended_stats.py`，计划 `hpc/fig2_extended_stats_plan_replicates_20261017.json`，作业 25852384，命名空间 extended_stats_v1，输出根独立于正式表）。归档：`runs_export/fig2_v37_final_20261014/runs/fig2_gptsub_b288_v37r2/`（4,954 文件 md5 核对）、`extended_stats_replicates/`、`tables/fig2_extended_stats_replicates.csv`、README 第 8 条。

## 12. v3.8：变体代 + 端点族规则（2026-10-17，先在 GPT b320/b360 验证）

复跑（§11）表明 GPT 在 b320/b360 的低值是这条臂在 v3.7 下的固定行为：三档提案 BB 占比 7–12 %（Claude/DeepSeek 37–59 %），
编排器自报 exploit 6 %、novelty 78 % 固定不变，结构端点名额反复给 HGP 垃圾码（b360 复跑 32 个端点名额里 21 个给 HGP、16 个
测出 ≥ 1e-4），自己提的 BB 参数也差。用户决定把两件 v3.7 留给模型自己做的事改成机器规则，三臂同一规则（commit 2d55f13）：

| 规则 | 内容 | 依据 |
|---|---|---|
| 变体代 `exploit_generations = 2` | 每轮 4 代里后 2 代只允许对父代做参数变体：父代 = 名人堂按实测 plpr（每权重类 ≤ 2，最多 5 个）+ 本预算点的常驻参考码（BB144 / BB72）。提示词只给这些父代，按 P<index> 指代、不点族名、不给目标值；fast loop 只接受父代所属族的程序，其余记 `family_not_allowed` | 进化搜索的探索/利用比例由框架定（FunSearch 一类系统的做法），不交给模型；三档赢家全是 BB144 附近的 BB 变体 |
| 端点不同族 `endpoint_family_distinct` | 第二个结构端点若与第一个同族且 (k, d_search, −w) 前沿上有别的族，取别族的最大 d_search 行 | 只改变第二个名额证明哪个前沿点，不放进被支配码 |
| 端点降级 `endpoint_demotion_strikes = 3, factor = 10` | 某族最近 3 次结构端点测量全部失败（未测出 / 被 10 倍规则淘汰 / plpr ≥ 10 × 最好实测码含参考码），该族此后不再占结构名额；终局名单同规则 | 与 racing 规则同一标准，族中立，Claude/DeepSeek 正式运行每个 run 只有 1–7 个这样的名额 |

协议身份 `telescoping_structural_v3_8`（快照 `structural_retention` 带四个新字段；v3.7 标签若带任一规则会被重标为变体）。
入口：`run-campaign --protocol-profile v3_8`、launcher `PROTOCOL_PROFILE=v3_8`（须 VALIDATE_FORMAL=0）、验证器 `--profile v3_8`。
部署：一棵独立的部署树 = Fig.2 三档复跑的代码基础 + v3.8 补丁（**不含** v3.7.1 的跨轮权重归并修复 e30746c，使 v3.8 与 Fig.2 v3.7 运行只差上述两条规则）。
运行：GPT b320 作业 25849063 起、GPT b360 25849065 起（RUN_TAG=v38，127 核 q3，gate w ≤ 8、30 次早停、搜索看门狗 8 h、终局无上限与三档复跑相同；10-17 12:43 启动）。10-17 16:40 用户决定再开 GPT b288（25852243 起 2 节）与 DeepSeek b320（25852245 起 2 节），同树同参数（DeepSeek 不带 QUOTA_MAX_WAIT_S），目录 `fig2_gptsub_b288_v38`、`fig2_deepseek_b320_v38`；同点的 v3.7 复跑继续跑完，最后两种协议的结果并列比较，入图仍受本节"三臂同规则"约束。
**GPT v3.8 结果（10-18 19:35 更新，三点补测完成）**：三条 30 轮完成，`validate_fig2_campaign.py --profile v3_8` 全部通过。终局冠军都是 BB（变体代从 BB144 参考码派生）。补测按点统一发数（同 §6 第 3 轮），计划 `hpc/fig2_extended_stats_plan_v38_20261018.json`，作业 25864474–476，输出在 NSCC scratch。

| 点 | 冠军 | 终局 25k/基 | 补测 | v3.8 pooled | 正式 v3.7 pooled | v3.7 复跑 pooled | 线值（60–260 拟合外推） |
|---|---|---|---|---|---|---|---|
| b288 | e5146726 [[144,4,14]] BB w6，T=14 | 2+1，2.14e-6 | 500k/基 36+25，2.18e-6 | **2.18e-6 [1.55e-6, 3.07e-6]** | 6.03e-6 | 3.61e-6 | 2.58e-6 |
| b320 | c9334f1c [[154,6,12]] BB w6，T=12 | 1+1，1.11e-6 | 850k/基 71+128，3.25e-6 | **3.19e-6 [2.63e-6, 3.87e-6]** | 4.86e-6 | —（22 轮停） | 1.02e-6 |
| b360 | e4062596 [[180,4,16]] BB w6，T=16 | 0+0 | 1.7M/基 45+27，6.62e-7 | **6.52e-7 [4.73e-7, 9.01e-7]** | 3.37e-6 | —（21 轮停） | 3.21e-7 |

相对线值：b288 0.84 倍、b320 3.1 倍、b360 2.0 倍（v3.7 最好分别 1.4、4.8、10.5 倍）。b360 的 Z 基补测只有 27 次失败，低于 §6 的 30 次目标；按点统一发数规则不再单独加测，区间 1.9 倍宽，与其他冠军相当。本地合并表 `runs_export/fig2_v38_variant_20261018/tables/fig2_extended_stats_v38.csv`。b320 冠军的终局值回归了 2.9 倍（终局只有 2 次失败）；同一终局里另两个 [[154,6,14]] BB 码 4+0、1+4 次，冠军规则不允许事后换码。DeepSeek b320 v3.8 搜索 30 轮完成，终局测完 5/7 时按用户要求停止，最好 [[140,6,14]] BB 3.81e-6（不如 v3.7 第二次运行的 1.61e-6，未补测、未进图）；前 10 轮 47 % 慢环名额 decode_infeasible（v3.7 复跑 25 %），DeepSeek 的变体代把 BB144 往 d ≥ 14 推，撞解码代价上限；10-18 第 22 轮因 NSCC 家目录配额满（Errno 122）停约 1 小时，腾出空间后自动续跑。归档（本地）：`runs_export/fig2_v38_variant_20261018/`（三条运行 15,009 文件 + 补测 24 文件，md5 核对）。

**论文约束**：v3.8 的点只有在三臂三档都按 v3.8 跑过之后才能进图（否则是给 GPT 开小灶）；用户定案的顺序是先看 GPT b320/b360 是否上去，上去了再跑 Claude/DeepSeek 三档 6 条和 GPT b288。v3.7 的 GPT b320/b360 复跑在 22/21 轮时按用户要求停止，checkpoint 保留。

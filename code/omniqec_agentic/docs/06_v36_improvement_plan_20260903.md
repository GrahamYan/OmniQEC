# 06 改进方案 `telescoping_structural_v3_6`（2026-09-03；decoder 定案见 §1 与 §7 第 6 条）

> 四项目标（用户原话归纳）：① 用新 decoder 替换冻结 BPOSD，其它配置不变；② detector 两基；③ 清除所有可能误杀大 d / 大 k 好码的机制；④ 让 generator 获知"d 才贵、大 N 要偏非阿贝尔"，并确保非阿贝尔码拿到合理的 SEC 和合理的解码。
> 证据来源：`05_large_code_fairness_audit_20260903.md`（审计与实测），`tmp/audit_20260903_large_code_fairness/`（数据与原型代码）。权威顺序仍是代码 > 04 > 本稿。

## 0. 不变项与变更项

**不变**：噪声 `uniform_paper_v1`、p=0.5% 单点；T=12；SEC 电路 `twobasis_nonint_v1`（两基不交错，X 半周期后 Z 半周期）；detector 两基都发；search 20,000 shots/基、final 200,000 shots/基、16/128 shards、racing 关闭；结构 (k, d_search) 保留通道与 3S+1R 慢位、8S:4R 父代、fixed-T 终局；纵轴 plpr = 1 − (1 − block)^(1/(kT))；prompt 三条约束（目标不进 prompt、不 warm-start、已知码表只判重发现）。

**变更**：decoder；协议 id `bposd_structural_v3_5` → **`telescoping_structural_v3_6`**（同日更早的自研级联草案 `cascade_structural_v3_6` 退役、从未上机；旧 run 数据不可合池，需重跑三臂）；§3 的机制修复；§4 的机器规则 / 种子 / coverage / SEC 认证与候选。

## 1. Decoder：`telescoping_cpu` 替换 BPOSD（用户 2026-09-03 定案：换成论文 telescoping 的源码，其余不动或按其适配）

### 1.1 算法定义（协议的一部分，全部字段进快照）

正式 decoder 是 Mitten 论文开源的 telescoping decoder（arXiv:2607.28795 App. I.1；github.com/a7b/yarn，MIT；vendored 在 `third_party/telescoping-decoder`，commit `c97099b759dcb53520d0ce210c6893dd9e0e0ea3` = 协议身份，import 名 `telescoping_decoder`）的 **CPU 配置**，preset `telescoping_cpu_v1`（`config.TelescopingDecoderConfig`）。同日更早的自研级联草案 `relay_cascade_gari_v1`（S1 BP → S2/S3 Relay-BP quorum → S4 HiGHS → S5 计失败；05 §9.2）已删除，从未上机。

| 级 | 状态 | 说明 |
|---|---|---|
| S1 flooding BP、S2 GPU relay-BP | **关** | 需要 CUDA；NSCC 与本机都是 CPU |
| S3-A | 开 | 独立 variant 集成（包默认表 5 个 variant），`a_iters=100`、`a_alpha_max=0.99`、`a_tau=5.0` |
| S3-B | 开 | relay-BP 带记忆 + coset quorum（包默认表 6 个先验 variant）：`b_runs=40`、`b_iters=100`、`b_alpha_max=0.90`、`b_use_memory=True`、`b_gamma0=0.1`、`b_gamma0_iters=80`、`b_gamma_center=0.21`、`b_gamma_width=0.90`、`b_stop_nconv=5`、`b_quorum=3`、`b_min_cost_select=False` |
| S3-C | 开 | 顺序参数扫描（包默认表 12 个配置，11 个含先验噪声各 `c_n_seeds_default=10` seed、1 个无噪 1 seed），`c_tau=5.0` |
| S4 Gurobi 整数规划认证 | **关** | 无 license；CPU-only 协议 |
| NC（无级接受） | **计为逻辑错误** | `nc_policy="fail"`：论文对未解出 shot 的规则；对所有码同一规则，LER 是上界；不再调 BPOSD，没有任何兜底（旧稿的 S5 / `fallback` 概念取消） |

输入：每个 memory 基的两基 DEM + 发射器记录的每 detector 基标签（`MemoryExperiment.detector_kinds`，随 DEM bundle 落盘，不在 worker 里解析电路）；`s3_system="gari"`——S3 在包自己的 GARI 变换系统上跑（`variant_tables="package_default"`，从不改包的表）。两基的 telescoping 系统（GARI npz）在 DEM build 时**物化一次**到 bundle 旁（`verification/decoders.py::materialize_telescoping_artifacts`；BB144 每基约 4 s、Mitten200 约 10 s），decode shard 只加载，并交叉核对发射器 detector 种类与包的 mask、系统的 detector / observable 计数与矩阵。`n_procs=1`：并行由 slow pool 拥有（每 shard 一核），包的 S3 pool 是每 shard 一个子进程。全部字段都是协议身份（改任一项须换 profile id），与预算点无关（`TELESCOPING_PROTOCOL_FIELDS = TelescopingDecoderConfig().to_dict()`）。

确定性：包从 `(base_seed, shot id)` 派生每个 S3 seed；适配层以 shard seed 为 `base_seed`、shard 内 shot 序号为 id，解码是 (矩阵, shard seed, shot 序号) 的纯函数——跨机器、pool 宽度、重跑逐 shot 一致（测试断言）。
记录：每 shard 输出 `stage_counts` / `stage_failures` / `stage_time_s`（键 `S3A` / `S3B` / `S3C` / `NC`）、`declared_failures`（= NC 数）、`nc_policy`、包的 per-shot 标签 `label_counts` 与 `gari{n_detectors,n_obs,shape,nnz}` → 进 slow 证据与 ledger（§4.5 的公平性可审计项）。

实测（p=0.5%，两基电路，T=12；05 §9.3）：BB144 X 1/2000、Z 0/2000（扩展 13/24000），0.07 CPU-s/shot；Mitten200 每基 1/60，11–18 CPU-s/shot；对冻结 BPOSD 的 2.40%/10.2% 与 92%/100%，以及自研级联草案的 0.50%/0.20% 与 47%/50%（130–143 s/shot）。

### 1.2 接入点（文件）

| 文件 | 改动 |
|---|---|
| `third_party/telescoping-decoder/`（新，vendored） | 论文源码，`VENDORED_COMMIT` 钉 commit；`pip install -e` 安装（editable；launcher 自检按包旁 `VENDORED_COMMIT` 核对）；两个 C kernel（`checkserial_bp` / `relay_mem_bp`）由 `python -m telescoping_decoder.build` 预编译进 `$XDG_CACHE_HOME/telescoping_decoder`（gcc；`CHECKSERIAL_BP_SO` / `RELAY_MEM_BP_SO` 可指向预编译件） |
| `config.py` | `DecoderName.TELESCOPING = "telescoping_cpu"`（CASCADE 删除）；`TelescopingDecoderConfig`（上表全部字段，`to_dict/from_dict/to_telescope_config/worst_case_iterations`）；`DecoderBuildConfig.telescoping`（blob 键 `telescoping`，`cascade` 键删除）；`TELESCOPING_PROTOCOL_FIELDS`；`budget_policy` 只剩 `{large_budget, search_shots, final_shots}`（`s4_time_*` 键与 `LARGE_BUDGET_S4_TIME_*` 常量删除）；`TELESCOPING_COST_S_PER_COLUMN_ITER` / `TELESCOPING_GARI_COLUMN_FACTOR` / `TELESCOPING_S3_WORST_CASE_ITERATIONS_DEFAULT` / `TELESCOPING_PACKAGE` / `TELESCOPING_VENDORED_COMMIT`；`StructuralSearchConfig.profile_id = "telescoping_structural_v3_6"`；`VerificationConfig.decoders = (TELESCOPING,)` |
| `verification/decoders.py` | `TelescopingObservableDecoder`（`telescoping_decoder.TelescopingDecoder` 的薄适配层；`Stage.NC` → `last_declared_failures`；`close()` 释放子进程）；`build_observable_decoder(..., detector_kinds=, basis=, stage=, seed=, circuit_text=, telescoping_artifacts=, workdir=)`；`materialize_telescoping_artifacts` / `resolve_telescoping_artifacts` / `build_decoding_system` / `is_x_detector_mask`；`decoder_dependency_report()` 版本含 `telescoping_decoder` / `telescoping_commit`。`verification/gari.py` 删除（`detector_kind_counts` 移到 `verification/circuit.py`） |
| `sec/stim_builder.py`、`verification/circuit.py` | 不变：`_emit_circuit` 记录每个 DETECTOR 的 kind；`MemoryExperiment.detector_kinds` / `to_metadata` |
| `evaluation/dem_cache.py`、`evaluation/worker.py` | bundle 带 `telescoping` 键（两基 GARI npz 相对路径、形状、nnz、指纹、build 秒数；旧 bundle 为 None）；build_dem 物化系统（失败 = `build_error`）；shard_dem 传 circuit_text + artifacts 并关闭 decoder |
| `evaluation/pipeline.py`、`evaluation/common.py`、`evaluation/writeback.py` | 进程内路径同样传 circuit_text 并关闭；`telescoping=TelescopingDecoderConfig.from_dict(...)`；`DECODER_BUILD_KEYS[telescoping_cpu] = "telescoping"` |
| `evaluation/pool.py` | `telescoping_worst_case_shot_s` 替代 cascade 版；`decode_budget_model` 键 `telescoping_worst_case_shot_s` / `telescoping_term_s` / `cost_s_per_column_iter` / `gari_column_factor` |
| `workflow/state.py`、`workflow/checkpoint.py`、`hpc/validate_fig2_campaign.py` | `PROTOCOL_PROFILE = telescoping_structural_v3_6`；`cascade_structural` 与 `bposd_structural` 同为退役前缀（resume 按名拒绝）；`_VERSIONED_PACKAGES` += `telescoping-decoder`，`resolved_versions` 加 `telescoping_commit`；validator 钉 `decoders == ["telescoping_cpu"]`、每个 `TELESCOPING_PROTOCOL_FIELDS` 键（缺一即失败）、`budget_policy` 三键、`resolved_versions.json` 含 `telescoping-decoder`（版本串）与 `telescoping_commit == TELESCOPING_VENDORED_COMMIT`（ldpc / relay_bp 不再要求） |
| `qualification.py`、`docs/qualification/` | 参照 decoder 改为 `telescoping_cpu`；集合含 Mitten150/200（`qualification_set_v1.json` v1.1） |
| `tests/test_telescoping_decoder.py`（新，替代 `test_cascade_decoder.py`） | 适配层：detector 种类 / 系统计数交叉检查、NC 计失败与 `declared_failures`、同 seed 确定性、级别统计、artifacts 加载路径 |
| `hpc/nscc_claude_grid_v36.pbs`、`submit_claude_v36_grid.sh`、`claude_v36_status.sh`、runbook v36、`hpc/requirements_fig2_nscc.txt`、`requirements.txt`、`hpc/README_FIG2_NSCC.md` | `DECODERS="telescoping_cpu"`；`module load gcc` + editable 安装 + 序言 `export XDG_CACHE_HOME=runs/.cache` 与 `python -m telescoping_decoder.build` 预编译（worker 继承，不编译；失败 exit 4）；`V36_DEFAULTS_OK` 断言 profile / decoder / `TELESCOPING_PROTOCOL_FIELDS` / 三键策略 / vendored commit；状态视图显示 `telescoping` preset；rsync 必须带 `third_party/telescoping-decoder` |
| `docs/02` §11.4、`docs/04` 追加（八）、`docs/05` §9.3 | 定案记录 |

### 1.3 资格测试与验收（本机，p=0.5%，两基电路；用户要求精简）

集合：BB72、BB144、Mitten150、Mitten200；BB 类 1000 shots/基，Mitten 100 shots/基（`qualification_set_v1.json` v1.1）。
验收（精神不变，判据是收敛与错误率，不是哪族赢）：(a) BB144 X/Z ≤ 0.5%；(b) Mitten200 ≤ 50%（实测 ≈2%）且 `declared_failures == stage_counts["NC"]`（失败只来自"无级接受"，没有隐藏的兜底）；(c) BB72 < BB144 保持；(d) 两次同 seed 逐 shot 一致。
测试只保留 4 项：detector 种类 / 系统计数交叉检查（BB144、Mitten200）、NC 计失败与 `declared_failures`、同 seed 确定性、一个端到端 shard + validator 冻结字段。

### 1.4 成本与按预算点放宽（用户 2026-09-03 决定：大码放宽，按预算点而非按码）

BB144 约 0.07 CPU-s/shot；Mitten200 约 11–18 CPU-s/shot（本机实测；NSCC 核更慢，以 canary 重估）。同一预算点上所有码同一套 shots（公平），不同预算点可不同；**没有逐级时限**（S3 各级按迭代数封顶，`budget_policy` 只含 shots）：

| 预算点 | search shots/基 | final shots/基 | decode watchdog（下限 + 模型） |
|---|---|---|---|
| ≤ 288 | 20,000 | 200,000 | max(1800 s, 8 s/shot × shots × max(1, DEM 列数 / 60,000), shots × 最坏一枪) |
| ≥ 300 | 5,000 | 50,000 | 同上 |

最坏一枪（每级都耗尽）是成本模型而非墙钟：`TELESCOPING_COST_S_PER_COLUMN_ITER`(2e-8 s，Apple M 系实测 × 2 余量) × `TELESCOPING_GARI_COLUMN_FACTOR`(1.3) × 两基 DEM 列数 × `worst_case_iterations()`（包默认表：5×100 + 6×40×100 + S3-C 各配置 = 104,000 次 BP 迭代）——Mitten200（123,960 列）≈ 335 s，BB144（67,104 列）≈ 181 s。真实成本由 shots 决定：b360 每候选 search 5k × 2 基 × 11–18 s / 128 核 ≈ 0.3–0.4 h，一轮 4 候选 ≈ 1.5 h；终局 50k × 2 基 ≈ 2.5–4 h/候选。快照键 `n_budget` / `budget_policy`（三键）/ `decoder` / `telescoping` 进 validator。

## 2. detector 两基（已满足，只做确认）

电路仍发两基 detector（`audit_twobasis_static` 计数不变）；decoder 经 GARI 消费两基 DEM；导出/硬件阶段不变。删除 05 §4 的 B1（同基投影）建议——被 §1 取代。B2（锚点改 0.2%）同样取消：新 decoder 让 0.5% 可测，验收 (b) 是硬门。

## 3. 清除误杀机制（05 §4 清单，逐条落地）

| # | 改动 | 文件 | 测试 |
|---|---|---|---|
| B3 | 近重复门改为 `code_hash` + pynauty `tier2_canonical_hash`（无 pynauty 时不按 WL 门）；结构行持久化 `canonical_hash` | `campaign.py:352-394`、`memory/archive.py:957-984,1025-1050`、`verification/structural.py`（metrics） | 两个不同 C4xD5 1x2 码同 turn 都可入队；测过一个后另一个仍可入队 |
| B4 | reducer 每 (k, d_search) 坐标 ≤2 个实现；slot 3 跳过已选坐标；父代按唯一坐标轮转（S,S,R）；`reducer_id` bump | `ranking.py:553-565,632-635`、`memory/archive.py` 父代、`workflow/graph.py:757-762` | 20×(45,4)+20×(4,15)+(40,12) → (40,12) 在 elite、slot 3、父代窗口 |
| B5 | `decode_infeasible` 改为终局可重试一次（128 shards）；预算随 DEM 列数缩放（§1.2） | `evaluation/pool.py`、`workflow/graph.py:989-1003,1291-1311` | 超时候选进入终局重试 |
| M1 | 层序认证：迭代数 = max(400, 4·核维)，≥3 seed 取 min，top-4 复认证，d_ext ≤ d_search 钳制，未收敛标记 | `sec/twobasis_rank.py`、`config.SecPolicyConfig`（新字段 sec_cert_seeds、sec_refine_iters） | Mitten150 d_ext = 7/7、BB144 10/9 复现 |
| M2 | `degraded_ext` 只作审计标志，不再降 tier / 踢 HoF、父代；`schedule_catastrophic` 仍是唯一硬隔离 | `ranking.py:159-171,215`、`memory/archive.py:759,784`、`evaluation/evidence.py:166` | 测过的 degraded 行排在未测行前 |
| M3 | `MAX_GROUP_ORDER` 由预算导出（n_budget // 4），常量改资源护栏 200；上限写进 contract 行；加 failure action | `config.py:84`、`groups/catalog.py:183-189`、`construction/registry.py`、`funsearch.py` | b360 下 D45 可构建；b288 下被拒并给出提示 |
| M4 | worker 数按 RSS 上限（记录 ru_maxrss） | `evaluation/pool.py`、pbs | — |
| M5 | 先提交 noise 0 的 build，落地后再提交 distance_job 与其余 build（或 synthesis 文件锁） | `evaluation/pool.py:1255-1274`、`verification/circuit.py:233-256` | synthesis 只算一次 |
| 次要 | b≥288 注入 BB144 resident baseline；ledger scope 用 n_total；known_codes 加 Mitten 150/200；`d_hint` 出 synth 元数据；目录同构去重（阶直方图 + \|Z\|） | `baseline.py:40-44`、`ledger.py:90-94`、`docs/known_codes_v1.json`、`circuit.py:215`、`groups/catalog.py` | 各一条 |

## 4. generator 信息与非阿贝尔码的 SEC / 解码保障

4.1 **机器规则**（= B4）：父代按唯一 (k, d_search) 坐标轮转后，同 k 下不同 d 的码自然并列出现，模型无需被告知"d 才稀缺"。
4.2 **种子：默认不加**（用户 2026-09-03 顾虑：1×2 直积群正是 Mitten 的结构要点，加种子有"引导"暴露面）。第一轮 campaign 只靠 4.1 + 4.3 的机器规则。campaign 后若 ledger 显示 1×2 非阿贝尔区域"从未被提出"，作为预注册的第二轮变更加入：用固定 RNG 按每个语法分支各生成一个平庸种子（生成脚本入库、d_search ≤ 6、预算门控），不手挑，三臂一致。
4.3 **coverage 仪表**：`construction/structure.py::structure_key` 加 (shape_a, shape_b) 与预算利用率分箱；`memory/coverage.py::catalog_gaps` 输出"本预算下从未提出的（分支 × 形状）"与每区 best_k / best_d_search；以事实行进策略简报与 Policy S-9，无 maximize/prefer 动词（`test_prompt_neutrality` 不改）。
4.4 **SEC 保障**：(a) M1 认证修复；(b) **群元素层候选**：对 LP 家族，把 `layer_blocks` 的项按 Mitten 方式分成左正则段与右正则段，各段内排列作为第二类候选（K 一半给着色排列、一半给分段排列），按修复后的 d_ext 取 min 最大者。证据：Mitten200 我们 Z 侧 d_circ ≤9、论文 12/12。可放第二阶段。
4.5 **解码保障**：§1 的 `telescoping_cpu`（资格测试含 Mitten150/200）；ledger 记录每码各级与 NC 的比例，任何"因 decoder 放弃而测差"的码可审计；可选第二阶段：每码固定预算自动调 γ 区间（4 组 × 50 个 pilot shot，按 S2 收敛率选，写进快照），所有码同一程序。
4.6 **文献事实块**：暂不上（进 prompt 文字、族不对称）；若 4.1–4.3 后 ledger 仍显示 1x2 直积分支"提出但从不测"，再考虑，并进 `prompt_ids()` 版本化。

## 5. 实施顺序与里程碑

1. **M0 decoder（2–3 天）**：§1.2 全部接入 + 测试 + 本机资格测试 §1.3。
2. **M1 机制修复（1.5 天）**：B3、B4、M2、M3、次要项 + 测试。
3. **M2 SEC 与执行几何（1.5 天）**：M1 认证、B5、M4、M5；4.4(b) 分段候选若时间允许。
4. **M3 generator（1 天）**：4.2 种子、4.3 coverage、父代轮转联调、中立性测试。
5. **M4 协议与上机（1 天）**：profile bump、validator、pbs/runbook、docs 02/04/05/06；全量 pytest（基线 417 + 新增）；本机 b144 两轮 smoke campaign；NSCC pytest → canary → grid。

每步验收：pytest 绿 + 对应的 §1.3 / §3 测试；M0 后本机复测 BB144 与 Mitten200 三码作为回归基线。

## 6. 可辩护性（"引导 LLM"质疑的回应口径）

- §3 全部是族无关 bug，审计文档有可复现失败案例，且同样误杀 BB / hgp（WL 门把 BB144 与 k=4 的 BB(12,6) 判同一码；端点重复挤出任何族的中间 Pareto 点）——修测量仪器。
- decoder 更换同理：旧 decoder 对权重 9 码 100% 失败 = 测不了；新 decoder 在固定参照集（各族文献码）上资格测试，判据是收敛与错误率，不是哪族赢；decoder 本身是论文的开源实现、commit 钉定、全部字段进快照；每 shard 记录各级与 NC 的计数，"因 decoder 放弃而测差"可审计。
- §4.1 / 4.3 是族无关探索规则：父代按唯一 (k, d_search) 坐标轮转；coverage 只报"本预算下没试过的区域"，bb 的 2×2 空洞同样报；无 maximize/prefer 词，中立性测试不变。
- §4.2 种子与 4.6 事实块是仅有的两处会被质疑的"先验注入"，默认都不上。
- 论文表述：所有协议在快照里预冻结、三臂完全一致、prompt 无目标词、已知码表判重发现、decoder 在固定参照集资格测试——发现是搜索结果而非提示结果。

## 7. 风险与待拍板

1. S5 "计为失败"而非 BPOSD 兜底（建议：是）。
2. 按预算点放宽（§1.4 表）：b≥300 用 5k / 50k shots 与 30 s / 300 s S4 时限（已按用户意见写入，待确认数字）。
3. 4.4(b) 分段层候选与 4.5 自动调参放第二阶段（建议：先不做，M0–M4 完成后按资格测试数据决定）。
4. 三臂旧数据与 v3.6 不可比，需重跑。
5. HiGHS 多线程/确定性需在测试中验证；不满足则 S4 改固定 node_limit。（随第 6 条作废：没有 HiGHS 级了。）
6. **2026-09-03 用户定案：decoder 换成 telescoping 源码（CPU 配置），其余不动或按其适配。** §1 已按此改写：正式 decoder = vendored `telescoping-decoder` 的 `telescoping_cpu_v1`（S1/S2/S4 关、S3-A/B/C 开、NC 计失败）；第 1 条的 S5 / 兜底概念取消（NC 即失败）；第 2 条的 S4 时限随之作废（shots 数字 5k/50k 不变，`budget_policy` 只剩三键）；§8 第 1–4 行的 cascade 特有修复随 cascade 代码一并退役（看门狗"下限 + 最坏一枪模型"的结构保留，常数换成 telescoping 的）。协议 id `telescoping_structural_v3_6`。

## 8. 审阅修复（2026-09-03 晚，六项已验证发现，代码 > 本节）

> 本节记录的是自研级联草案（`relay_cascade_gari_v1`）时期的审阅修复，保留为历史。第 1–4 行的 cascade 特有项（S4 时限、`cascade` 键、`_cascade_config_ok`）随 decoder 换成 telescoping（§7 第 6 条）一并退役——第 1 行的 `stage` 贯穿与第 2 行的看门狗结构（下限 + 最坏一枪模型）保留，第 3 行的 `DECODER_BUILD_KEYS` 现映射 `telescoping_cpu → "telescoping"`，第 4 行的 resume 规则现按退役前缀 `bposd_structural` / `cascade_structural` 或缺 `budget_policy` / `telescoping` 键拒绝；第 5–6 行与 decoder 无关，仍有效。

| # | 发现 | 修复（根因处） | 测试 |
|---|---|---|---|
| 1 | 串行/进程内慢路径写死 `stage="search"`，终局落到 workers=1 时用 search 的 S4 时限并记 `decode_stage="search"` | `stage` 贯穿 `evaluate_slow_spec` → 子进程请求 → worker 默认模式 → `evaluate_slow_spec_inprocess`；`pool._evaluate_one` 与 job 路径同规则（终局或 `retry_at_final` → final） | `test_v36_review_fixes.py`：BB72 进程内终局 `s4_time_limit_s == 600` |
| 2 | decode 看门狗 8 s/shot × 列数因子（Mitten200 = 16.5 s/shot）远低于离开 S1 的一枪实测 ~250 s（S2 46 + S3 144 + S4 60），Mitten 类 shard 会被判 `decode_infeasible` | 看门狗 = max(v3.4 项, shots × **cascade 最坏一枪模型**)：`1.2e-7 s × 列数 × (s1+s2_sets·s2_iters+s3_sets·s3_iters=26200) + 该阶段 τ4`（实测 4.3–9.2e-8，取 2 倍余量）。不封 S2/S3 墙钟：那会破坏同 seed 逐 shot 一致。`job_config.decode_budget_model` 记录两项 | 同上：312 shots × 50% × 250 s 在看门狗内；列数未知时与 v3.4 完全一致 |
| 3 | `writeback` 按 decoder **名字** 查 build blob，v3.6 名字 `relay_cascade_gari` ≠ 键 `cascade`，每行 `decoder_config=None`，终局严格审计与 validator 必挂 | `DECODER_BUILD_KEYS` 名→键映射 + `decoder_build_blob()` | cascade 行 blob == 快照 cascade blob；`_cascade_config_ok` / validator 通过；真实 smoke 产物复核 |
| 4 | `assert_resumable_protocol` 把任何非 cascade decoder 都当 v3.5 检查点拒绝，非严格 `+declsd` 变体无法续跑自己 | 只按退役 profile 前缀或缺 `budget_policy`/`cascade` 键拒绝；变体相同性交给 `validate_resume_protocol` | `+declsd`/`+decbposd` 自续；v3.5 仍按名拒绝 |
| 5 | M1 认证 Mitten 类 [[200,40]] 需 ~340k ISD 迭代（本机 156–223 min）> build 看门狗 3600 s；synth cache 只在完成时写，重试从零开始 → 2 次 strike 出局 | ① `SecPolicyConfig.sec_cert_iters_budget`（默认 0 = 不封顶 = 本稿 M1 原样；协议字段、validator 钉 0、变体后缀 `B<n>`）：超预算时按固定整数规则缩 stage-1/refine 迭代并记 `certification_budget_exhausted`（仅审计，不是 strike）；② `CertificationProgress` 每个 probe 落盘 `synth_<key>.progress.json`，被杀后续算且数值逐位一致；③ build 看门狗 = max(3600, 认证工作量模型)：kernel≈4n、每迭代 3e-3 s×(n/72)^2.8（2 倍余量）+600 s → n=200 约 5.3 h，n≤100 仍 1 h | 计划函数、预算耗尽标志、断点续算、模型；pool 的 build 请求带 `n_data` |
| 6 | coverage "从未提出" 只枚举 LP 两族；bb/hgp 空洞永不出现；12 项截断每轮全是 lp_nonabelian/dihedral | 四族各有区域网格（bb：方/矩 × \|G\| 段；hgp：H1×H2 种子列数段），按族最小拼法判可行；`never_proposed_by_family` 计数；截断按族→分支轮转；中立性正则不变 | bb-only / lp-only 档案都报四族；12 项每族 3 条 |

实测（本机单线程，每 ISD 迭代）：BB72 1.5 ms、BB144 5.8 ms、Mitten150 12.8 ms、Mitten200 27.5 ms；完整 M1 = 3.4 / 21 / 57 / 156 min。位打包消元原型只有 2.1×（开销在逐 pivot 的 numpy 调用），故成本由 M1 迭代数决定。

**待用户拍板**：(a) b360 下 Mitten 类每 shot 均值实测 ~125 s（离开 S1 的一半 shot 各 ~250 s），一轮 4 候选 ≈ 11 h，§1.4 的 5 h 按 60 s/shot 估的；(b) 是否在 campaign 前设 `sec_cert_iters_budget`（≈60000 时 Mitten200 认证约 55 min/NSCC，但所有 Mitten 类码会带 exhausted 标志）；(c) NSCC 上按资格集实测 Mitten150/200 的 build 与 decode 时长再开 b≥300。

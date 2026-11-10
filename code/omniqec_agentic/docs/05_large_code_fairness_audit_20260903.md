# 05 大 d / 大 k / 非阿贝尔码公平性审计（2026-09-03）

> 目的：在 `bposd_structural_v3_5`（两基不交错 SEC `twobasis_nonint_v1` + 论文 SI C 噪声、p=0.5%、冻结 BPOSD product_sum/50/osd_cs/10、T=12）下，找出**所有**会把大码距、大 k 或非阿贝尔 LP 码误杀的机制，并实测 Mitten [[150,30,10]]、[[200,40,12]] 与 BB144 在当前默认协议下的读数。
> 方法：5 个只读 auditor（准入门槛 / 快环与结构保留 / 两基 SEC / 解码与测量 / 实测）+ 1 个 generator 信息方案设计 agent（工作流 `wf_a1cab552-db2`，Verify 阶段因 session limit 中断），随后本会话逐条在代码里核实每个引用位置，并补跑决定性实验。数据与脚本见 §8。

## 0. 结论

1. **v3.5 默认协议不能上 NSCC。** 在 p=0.5% 下，BB144 每基块错 33%，两个 Mitten 码 100% 失败；BPOSD 每 shot 17–200 秒（watchdog 8 秒/shot → 三个码全部 `decode_infeasible` 永久出局）；每 worker 4.4–9.4 GB（128 worker × 100 GB 节点 → OOM）。
2. 两个根因，都不是"两基 SEC"本身：
   - **锚点 p=0.5% 在两基全测电路下，冻结 BPOSD 对权重 9 / 度 12 的码饱和**。换成"decoder 只吃同基 detector"后 Mitten 仍 86–92% 失败；BPOSD 下 p=0.2–0.3% 才全部可测。这不是物理极限：论文 Fig.2 用 telescoping decoder 在 0.5% 处 Mitten 各码每轮 LER 在 1e-3–1e-2（12 轮块错百分之几到十几），阈值 p_c≈0.7%；见 §9 的 decoder 对照。
   - **交给 BPOSD 的 DEM 含两基 detector**（`sec/twobasis_builder.py` `detect_kinds={"X","Z"}`）：DEM 机制数 8 倍，p=0.5% 下 BP 不收敛、OSD 在 12 万列上跑 → 慢 200–500 倍、解得更差（BB144 33% → 2.4%/9.7%）、内存 25 倍。p=0.2% 下两者差距缩小（Mitten200 两基 DEM 13%/5% vs 同基 10%/5.7%），但同基 DEM 永远不劣且便宜 8 倍。
3. 与 SEC 无关、但同样会误杀大码的机制（全部在代码里核实）：WL 近重复门（同形状 LP/BB 码一个 campaign 只能测一个）、结构 elite/父代/终局被端点坐标重复实现占满、层序 ISD 认证在 n≥150 不收敛、`degraded_ext` 把测过的码排到未测码后面、`MAX_GROUP_ORDER=72` 在 b320/360 砍掉整个 1x1 非阿贝尔分支、`decode_infeasible` 终态。
4. 两基 + 冻结 BPOSD 下，权重 9 的 Mitten 码在 p=0.2% 比 BB144 差约 50–100 倍 plpr。这是电路距离（Mitten150 深度认证 7，论文自己也只有 8）与每比特噪声翻倍的真实代价，不是误杀。大 N 发现主轴应是结构 (k, d_search) frontier，运行时 plpr 作确认轴。

## 1. 实验设置

- 电路：`twobasis_nonint_v1`，K=16 / 400 ISD / seed 0 排名后的层序（与协议一致），`layer_blocks_v1` 结构层（Mitten 两码与 BB144 都走结构路径，每基 9 层 / 6 层）。
- 噪声：`uniform_paper_v1`，p ∈ {0.5%, 0.3%, 0.2%, 0.1%}（0.1% 只作斜率参考，用户已决定不作锚点）。
- 解码：冻结 BPOSD，含噪电路直接采样（协议采样源），OMP 单线程，Apple M4 Pro。
- "两基 DEM" = 当前默认（detector 两基都进 DEM）；"同基 DEM" = 同一电路，每个 memory 基只把该基 detector 交给 decoder（`_Timeline.detect_kinds={basis}`，物理完全不变）。
- Mitten 码：GAP SmallGroup(30,1)/(40,5) Table XIII 重建（仓库原生 spec 见记忆 `fig2-mitten-codes-under-v34-protocol`），k=30/40，权重 9，四 seed d_search=10/12。

## 2. 结果

### 2.1 当前默认（两基 DEM，p=0.5%，T=12，每基块错率）

| 码 | X | Z | 每 shot 秒 | worker 峰值内存 |
|---|---|---|---|---|
| BB144 [[144,12,12]] | 4/12 | 4/12 | 17–49 | 4.4 GB |
| Mitten [[150,30,10]] | 65/65 | 64/65 | 45–54 | — |
| Mitten [[200,40,12]] | 39/39 | 39/39 | 160–203 | 6.8–9.4 GB |

对照：旧 LRC 电路（同基 detector，无 idle）BB144 每 shot 0.2–0.3 秒、块错约 7%；组员论文 SI C 单基协议 p=0.5%：Mitten200 plpr 1.2e-5、Mitten150 4.5e-5、BB144 0/1040。

### 2.2 同基 DEM，p 扫描（每基 失败/shots）

| p | BB144 X | BB144 Z | M150 X | M150 Z | M200 X | M200 Z |
|---|---|---|---|---|---|---|
| 0.5% | 48/2000 (2.4%) | 194/2000 (9.7%) | 210/240 (88%) | 221/240 (92%) | 220/240 (92%) | 206/240 (86%) |
| 0.3% | 5/4000 (0.13%) | 29/4000 (0.73%) | 220/600 (37%) | 243/600 (41%) | 244/600 (41%) | 127/600 (21%) |
| 0.2% | 1/4000 | 3/4000 | 55/600 (9.2%) | 69/600 (11.5%) | 61/600 (10.2%) | 34/600 (5.7%) |
| 0.1% | 0/10000 | 0/10000 | 17/2000 (0.85%) | 34/2000 (1.7%) | 23/2000 (1.15%) | 7/2000 (0.35%) |

每 shot 秒：BB144 0.01–0.2，M150 0.05–0.44，M200 0.05–1.13（p 越低 BP 越常收敛、越快）。
两基 DEM 对照（M200，p=0.2%）：X 8/60、Z 3/60，0.2–0.3 秒/shot。

plpr = 1 − (1 − block)^(1/(kT))，p=0.2%：BB144 1.7e-6 / 5.2e-6；M150 2.7e-4 / 3.4e-4；M200 2.2e-4 / 1.2e-4。p=0.3%：BB144 8.7e-6 / 5.1e-5；M150 1.3e-3 / 1.4e-3；M200 1.1e-3 / 5.0e-4。
20k shots/基下的预期失败数：p=0.2% BB144 5–15、Mitten 1100–2300；p=0.3% BB144 25–145、Mitten 4200–8100。

### 2.3 层序认证的收敛性（`certify_basis_order`，5000 次 × 3 seed 取 min，对比协议的 400 次单 seed）

| 码 | 侧 | 协议 d_ext（400） | 深度 min | canonical 序深度 min |
|---|---|---|---|---|
| BB144 | X（选 idx 0） | 10 | 10 | — |
| BB144 | Z（选 idx 15） | 9 | 9 | 8 |
| M150 | X（选 idx 8） | 10 | **7** | 6 |
| M150 | Z（选 idx 2） | 10 | **7** | — |
| M200 | X（选 idx 1） | **19** | 12 | 7–8 |
| M200 | Z（选 idx 3） | 18 | **9**（9/10/12，未收敛，真值 ≤9） | 7 |

n=144 收敛，n≥150 不收敛：协议数字高于码距（19 > 12）说明 400 次是噪声；选"最大 d_ext"= 选噪声最大的候选（M200 X 选中 idx 1 是运气好）。Mitten 论文 Table VIII：[[200,40,12]] 电路距离 12、[[150,30,10]] 8，与深度认证一致。

## 3. 根因分析

- **p 锚点**：Mitten150 在 0.3%→0.2% 的斜率对应有效距离约 7（= 深度认证的电路距离），BB144 约 10；两者都在各自电路距离附近解码。差距来自 d_circ（7 vs 9–10）和每数据比特每轮的 CX 数（度 12 vs 6）。0.5% 对度 12 的码：每轮约 12 个故障对 d_circ 7，冻结 BPOSD 饱和；论文 telescoping 在同一 p 仍可测（§9），所以 p 固定时杠杆在 decoder。
- **DEM 两基 detector**：对基 detector 对 W-observable 只经 Y 关联提供信息；BPOSD 在 8 倍列数上 BP 不收敛、OSD 代价 O(m²n) 且解质量下降。Bravyi et al. 的 BP-OSD 也是分基解码。
- **M200 的 X/Z 不对称**（X-memory 更差）与 d_ext_z 侧对应：Z 侧深度认证 ≤9（9/10/12 未收敛）对 X 侧 12，论文两侧都是 12。我们的 9 层着色 + 16 个排列的候选空间在 Z 侧没找到保距离的序；论文是 12 层、左右正则分段、段内排列。

## 4. 机制清单

严重度：阻断 = 目标码在当前默认下无法被测/被排名；主要 = 系统性偏置；次要 = 局部或可绕过。每条已在代码里核实。

### 阻断

| # | 位置 | 机制 | 证据 | 修法 | 成本 |
|---|---|---|---|---|---|
| B1 | `sec/twobasis_builder.py:148` `detect_kinds={"X","Z"}` → DEM | decoder 吃两基 detector，DEM 8 倍 | §2.1 vs §2.2 | decoder 侧按 memory 基投影（电路仍发两基 detector 供导出/硬件）；新增协议字段 + profile bump | 0.5–1 天 |
| B2 | `workflow/state.py:37` `FORMAL_P=0.005`、`evaluation/sampling.py:22`、validator:66 | 两基电路下权重 9 码过阈值 | §2.2 | 锚点改 0.2%（备选 0.3%）；profile bump；重测 resident baseline | 0.5 天 + 重基线 |
| B3 | `memory/fingerprints.py:33` 1-WL；门在 `campaign.py:352-394`、`memory/archive.py:957-984, 1025-1050` | 同形状同权重 LP/BB 码同哈希；测过一个后同类永不进慢环 | 12 个不同 C4xD5 1x2 权重 9 码 → 1 个 wl_hash；BB144 与 k=4 BB(12,6) 同哈希；真实两码复现 selected=[]；canary ledger 里 (72,4,6)/(72,6,4)/(72,12,3) 同哈希 | 门改 code_hash + pynauty `tier2_canonical_hash`（已装、已实现、从未接入门）；无 pynauty 时不按 WL 门；结构行持久化 canonical_hash | 0.5–1 天 |
| B4 | `ranking.py:553-565` reducer、`:632-635` slot 3、`workflow/graph.py:757-762` 父代 | 端点坐标的重复实现（crowding_boundary 优先）占满 16 elite；slot 3 / 终局第三席 = 端点第二实现；父代窗口 4/6 同端点 | 20×(45,4)+20×(4,15)+(40,12) → (40,12) 出局；60 轮模拟 (40,12) 第 9 名，终局 [(48,8),(8,17),(48,8)] | 每 (k,d_search) 坐标最多 2 个实现；slot 3 跳过已选坐标；父代按唯一坐标轮转；`reducer_id` bump | 0.5 天 |
| B5 | `config.py:575` 8 s/shot；`evaluation/pool.py:1086-1089`；`graph.py:989-1003, 1291-1311` | decode_infeasible 终态、永不重试、不进终局；预算按 v3.4 LRC 校准 | §2.1 秒/shot | B1 后本机 0.02–1.1 s/shot 不再触发；仍应按 DEM 列数缩放预算、终局（128 shards）允许重试一次 | 0.5 天 |

### 主要

| # | 位置 | 机制 | 证据 | 修法 | 成本 |
|---|---|---|---|---|---|
| M1 | `sec/twobasis_rank.py:194-246` | 400 次单 seed ISD 在 840 列扩展核上不收敛；argmax 选噪声；d_ext > 码距 | §2.3 | 迭代数随核维缩放、≥3 seed 取 min、top-4 复认证、d_ext ≤ d_search 钳制、未收敛标记 | 0.5 天 + 每码约 20 分钟 |
| M2 | `evaluation/evidence.py:166`、`ranking.py:159-171, 215`、`archive.py:759, 784` | `degraded_ext`（probe_upper_bound 也算）→ tier 0、踢出 HoF/父代；BB144 自己就 degraded（10/9<12） | 代码核实 | 只作审计标志；`schedule_catastrophic` 是唯一硬隔离；不把测过的码排到未测码后 | 2 小时 |
| M3 | `config.py:84` `MAX_GROUP_ORDER=72`、`groups/catalog.py:183-189` | b320/360 上 \|G\|=73–90 的 1x1 非阿贝尔 LP 作 build_error 拒绝，prompt 未写上限，无 failure action | D40/D45/C5xD8/C9:C9(4) 实测被拒 | 由预算导出（n_budget//4），常量只作资源护栏（≈200）；上限写进 contract 行 | 0.5 天 |
| M4 | `hpc/nscc_claude_grid_v35.pbs:121, 592` | 128 worker × 4.4–9.4 GB → OOM SIGKILL → `measurement_blocked` → MAX_SLOW_FAILURES=2 出局 | run6 RSS | B1 后 <0.5 GB；仍按 RSS 上限限 worker 数并记录 ru_maxrss | 小 |
| M5 | `evaluation/pool.py:1255-1274`、`verification/circuit.py:233-256` | distance_job 与每噪声点 build_dem 同时提交，synthesis 并发重复 1+N 次（磁盘缓存完成后才写） | Mitten 一次 8 分钟；争用下 16–18 分钟 | 先提交 noise 0 的 build，落地后再提交其余；或 synthesis 文件锁 | 30 行 |

### 次要

- `evaluation/baseline.py:40-44`：b≠144/288 无 resident baseline → b360 无 BB144 对照（BB144 总量 288 装得下）。
- `memory/ledger.py:90-94`：scope 用 `n`（数据比特）而非 `n_total`，b≥288 的假设把 [[200,40,12]] 排除在外。
- `config.py:595` `d_cap=12`：只影响运行时 Q_eff 与 MAP，结构通道不受影响。
- `groups/catalog.py`：40 阶 11 个名字只覆盖 8 个同构类（C4:C10(3)=C20:C2(11)=C5xD4 等），Q8 类不可达；同构重拼写只在 WL 层去重（→ B3）。
- `docs/known_codes_v1.json`：无 Mitten 行，重发现会被记为新发现。
- `verification/circuit.py:215`：`d_hint` 不在 synth cache key，`schedule_degraded_x/z` 由第一个调用者决定（只影响信息字段）。
- `evaluation/pool.py:929-935`：max_errors 按 shard 切分（正式路径 max_errors=None，不受影响）。
- `sec/twobasis_rank.py:176`：`logicals()`/行约简在候选循环内重复算（M1 放大后才显著）。

### 核实为无问题

权重上限 `>` 9（权重 9 通过）；总量 = n + m_X + m_Z 与 prompt 一致；1x2 形状允许、cell 权重上限 8 不约束；Mitten spec 在 b360 合法可编译（k=40，四 seed d_search=12，trust filter 0.85 通过）；d_search 代理 n=200 上 30/30 与 16 seed×1000 次参照一致、6/6 与 MILP 一致，无 k 偏置；Pareto 只看 (k, d_search)，端点平手偏向另一目标；trust filter 只在运行时通道；plpr 的 1/(kT) 偏向高 k；racing 关闭；T=12 一致；catastrophic 判定不可能命中 d≥w 的码；结构层路径确实被 LP 使用；静态审计通过；文献拼写 C4xD10/C5xS3 明确指向目录名；known-code 查询不进任何排名路径。

## 5. generator 如何获知"k 便宜、d 贵、非阿贝尔解除阿贝尔上限"

事实基础：30 个随机 C4xD5 1x2/1x2 权重 9 抽样全部 k=40（k 由形状 × |G| 决定），d_search 分布 {4: 8, 6: 12, 7: 2, 8: 8}，只有文献码到 12；Mitten 论文 §III：同形状阿贝尔群 d ≤ 基矩阵校验权重（他们的 1x2 权重 9 对应"距离 6 上界"），非阿贝尔解除上限。

建议顺序（先修 §4 阻断项，否则任何往该分支的引导都会在当前协议下测成全场最差并被 ledger 证伪）：
1. **机器规则**（B4 修法本身）：elite 按坐标去重 + 父代按唯一 (k, d_search) 轮转 → 父代列表天然呈现"同 k 下 d 在变"，无需文字。
2. **种子**：`examples/seed_programs.py` 加一个 1x2/1x2 直积群平庸种子（d_search 4–6，按预算门控，b288/b360 各一）+ 一个 1x2 metacyclic 种子；先例是 metacyclic 种子；三臂一致、不进 prompt。
3. **coverage 仪表**：`memory/coverage.py::catalog_gaps` 扩到 (形状 A, 形状 B) × 预算利用率分箱，附 per-region best_k / best_d_search，以事实行进策略简报与 Policy S-9（无 maximize/prefer 动词，`test_prompt_neutrality` 可过）。
4. **文献事实块**（备选）：阿贝尔上限定理 + bb/hgp/lp_abelian 的对称已知上限，放 `orchestrator/prompts/generator_facts.md` 并进 `prompt_ids()`；只在 1–3 之后、且协议已能正确测量该分支时启用。
5. 不写"请生成非阿贝尔码"；已知码表仍只做重发现判定；不 warm-start。

## 6. 建议的协议版本 `bposd_structural_v3_6`（待用户确认）

| 项 | v3.5 | v3.6 提议 |
|---|---|---|
| SEC 电路 | twobasis_nonint_v1，detector 两基 | 不变 |
| decoder DEM | 两基 detector | **按 memory 基投影**（新字段 `decode_detector_scope`） |
| 锚点 p | 0.005 | **0.002**（备选 0.003） |
| 层序认证 | 400 次 × 1 seed | 迭代随核维缩放 + 3 seed min + 钳制 ≤ d_search |
| degraded_ext | tier 0 + 踢出 | 审计标志 |
| 近重复门 | WL | code_hash + pynauty canonical |
| 结构 reducer | 端点 + crowding | 每坐标 ≤2 实现，slot 3 唯一坐标，父代轮转 |
| MAX_GROUP_ORDER | 72 | n_budget//4 |
| decode 预算 | 8 s/shot | 按 DEM 列数缩放；终局可重试一次 |
| resident baseline | 144/288 | b≥288 注入 BB144 |
| known_codes | 4 条 | + Mitten [[150,30,10]] / [[200,40,12]] |

## 7. 需用户拍板

1. decoder DEM 只用同基 detector（建议：是）。
2. 锚点 p=0.2%（建议）还是 0.3%；0.5% 只能配单基 SI C 协议（已否决）。
3. 接受两基 + 冻结 BPOSD 下权重 9 码天然落后 BB144 一到两个量级；大 N 发现以结构 frontier 为主轴。若要运行时轴也对权重 9 码公平，约束是 decoder 冻结。

## 8. 数据与脚本

`omniqec_agentic/tmp/audit_20260903_large_code_fairness/`：
- `wonly.py`（§2.2 实验，`_Timeline` 猴补丁做同基投影）、`results.jsonl`、`run.log`
- `deepcert.py`、`deepcert_*.log`（§2.3）
- `result_mitten*_{blocks,fallback}.json`（§2.1，empirical auditor）
- `mitten{150,200}_reg_h{x,z}.npy`、`*_layer_blocks.json`、`registry_specs.json`、`g30_1.json`/`g40_5.json`（GAP 导出）、`build_mitten.py`
- `gates.json`、`fastloop.json`、`sec.json`、`decoder.json`、`generator_info.json`（五个 auditor 的结构化输出原文）

## 9. 补充（2026-09-03 下午）：decoder 对照 —— 冻结 BPOSD vs 论文 telescoping 的 CPU 复现，BB144，p=0.5%

用户要求用 Mitten 论文的 telescoping decoder 测 BB144（p=0.5%）与现有结果对比。论文的 CUDA/C kernel 未公开，按其 App. I.1 四级结构在 CPU 上复现（`tmp/audit_20260903_large_code_fairness/telescoping.py`）：S1 BP（ldpc product_sum 串行 100 轮）→ S2 Relay-BP（IBM `relay_bp` 0.2.1，README gross 码参数：gamma0 0.1、pre_iter 80、num_sets 60、set_max_iter 60、区间 (−0.24, 0.66)、stop_nconv 5）→ S3 Relay-BP 大集成（num_sets 400、gamma0 0.65、stop_nconv 3）→ S4 整数规划 MLE（scipy HiGHS 代替 Gurobi；先子 DEM 30 s，再全 DEM 60 s）→ S5 兜底 = 冻结 BPOSD。与论文的差别：全部级在**同基 DEM**上跑（论文 S2/S3 用全 XYZ DEM 的 GARI 变换）、无逻辑类 quorum 投票、S4 顺序与论文相反且时限远小于 1 CPU 小时。这些差别都只会让复现版**弱于**论文原版（Relay-BP 论文：XYZ DEM 在 gross 码 p=0.3% 再多约一个量级）。

同一两基不交错电路、同基 DEM、相同 stim seed 与分片、每基 20,000 shots（块错率 / plpr）：

| decoder | X | Z | 每 shot 秒 |
|---|---|---|---|
| 冻结 BPOSD product_sum/50/osd_cs/10 | 480/20000 = 2.40% / 1.7e-4 | 2047/20000 = 10.2% / 7.5e-4 | 0.10 / 0.40 |
| telescoping-CPU（同基、无 GARI、无 quorum） | 139/20000 = 0.70% / 4.9e-5 | 104/20000 = 0.52% / 3.6e-5 | 0.19 / 0.39 |

改善 X 3.5 倍、Z 20 倍；速度同量级。各级分布（X/Z）：S1 80%/75%、S2 19%/24%、S3 0.7%、S4 0.1%、落到 BPOSD 兜底 0/8 个 shot。失败按级：X S1 6、S2 91、S3 35、S4 7；Z S1 5、S2 54、S3 21、S4 17、S5 7 —— 绝大多数错在 S2/S3 接受了错误逻辑类，正是论文用 quorum 防的那一类。对照：Mitten200 在同一协议下用冻结 BPOSD 是 86%–92%（同基 DEM）/ 100%（两基 DEM）。

只读对抗审查（工作流 `wf_64171fb0-e67`）确认的偏差：(a) 无 quorum（major，失败主要来源）；(b) 同基 DEM 而非 XYZ+GARI（major，论文差距最大项）；(c) S4 子 DEM 先于全 DEM 且时限短（事实成立，对本次 40k 结果的影响 <1σ，被反驳为非 major）；(d) relay 实例未接 seed，单 shot 结果依赖同进程内解码顺序，shard 级可复现（minor）。数据：`tele_bb144_p005.jsonl`、`bposd_bb144_p005_20k.jsonl`。

### 9.1 Mitten [[200,40,12]]，p=0.5%：我们的冻结 BPOSD vs 论文 Fig.2（telescoping）

论文 Fig.2(a) 读图（300 dpi，纵轴 = 每轮块错率，X/Z 平均）：p=0.5% 时 [[150,30,10]] ≈ 5e-3、**[[200,40,12]] ≈ 1.5e-3**、[[300,60,14]] ≈ 6e-4；p=0.2% 时 [[200,40,12]] ≈ 1.2e-7；p=0.1% ≈ 6e-10。论文噪声 I.4：CX 后 DEPOLARIZE2、初始化后与测量前 DEPOLARIZE1、无 idle（与 SI C 只差测量前是去极化还是比特翻转）。

| 设置 | 每基 12 轮块错 X / Z | 每轮块错 | plpr | 每 shot |
|---|---|---|---|---|
| v3.5 默认：两基 SEC + 两基 DEM + 冻结 BPOSD | 39/39, 39/39 | ≥0.18（饱和下限） | ≈1（饱和） | 160–203 s |
| 两基 SEC + 同基 DEM + 冻结 BPOSD | 220/240 (92%), 206/240 (86%) | ≈0.17（饱和） | 4–5e-3 | 1.0–1.1 s |
| 论文：12+12 层 hook-free 调度（d_circ 12/12）+ telescoping + XYZ GARI DEM | — | ≈1.5e-3 | ≈4e-5 | GPU |
| 参考：组员论文 SI C 单基协议 + 冻结 BPOSD（2026-09-02 本机） | 2/1040, 4/1040 | ≈2.5e-4 | 1.2e-5 | — |

同一 p、同为两基全测，我们比论文差 ≥100 倍（我们已饱和，真实差距更大）。来源：decoder（BPOSD 在围长 4 的 mitten 图上比在 gross 上更弱；论文 Table IX 注）+ 调度（我们 Z 侧 d_circ ≤9 对论文 12）。单基 SI C 协议之所以在 0.5% 反而优于论文两基 telescoping，是因为单基测量门数减半且对基 hook 不存在。

### 9.2 新 decoder 原型 `relay_cascade_gari_v1`（2026-09-03 傍晚）：两基 DEM + GARI 变换 + 分级 BP，BB144 与 Mitten200 同配置对比

用户决定：detector 两基不动、decoder 也吃两基 DEM；按 Mitten 论文的分级逻辑做自己的 decoder。实测发现两基 XYZ DEM 上 Relay-BP 对 Mitten200 完全不收敛（100×60 不收敛 33–45 s，300×200 不收敛 279 s），原因是 GARI 论文（Maan 等，Nat. Commun. 2026，arXiv:2510.14060）指出的 Y 错误 4-环。按该论文 §II.A Eq.(5) 实现 GARI 变换（`tmp/audit_20260903_large_code_fairness/gari.py`，随机向量验证通过；Mitten200 X-memory：1920×123960、nnz 1.09M → 31760×153800、nnz 0.41M，即论文说的"边数约 1/3"；BB144 同理）。Mitten 论文自己的 telescoping 代码在 github.com/a7b/yarn（MIT），GARI 官方实现 astra-decoders/gari-nms（Apache-2.0），Google tesseract 也有 gari.py；我们的实现独立、按论文公式。

原型（`cascade_gari.py`）：S1 min-sum BP 带记忆（relay_bp，200 轮）→ S2 三副本 Relay-BP（100 集成 × 60 轮，gamma0 0.1，区间 (−0.24, 0.66)）逻辑类 quorum → S3 三副本（100 × 100）quorum → S4 兜底 = 冻结 BPOSD（原两基 DEM）。接受规则 = 答案侧 detector 行完全复现（论文 half-row）。参数未按码调；无整数规划级。

同一两基电路、同一 DEM、同一配置、每基块错：

| 码 | 冻结 BPOSD（两基 DEM，v3.5 默认） | 冻结 BPOSD（同基 DEM） | GARI cascade | 各级分布（cascade） |
|---|---|---|---|---|
| BB144 X | 4/12 (33%)，17–49 s/shot | 480/20000 = 2.40% | **10/2000 = 0.50%**（S3 300×200 时 5/2000 = 0.25%） | S1 96%，S2 3%，S3 0.3–0.6%，兜底 0.1–0.4%；0.6–0.8 s/shot |
| BB144 Z | 4/12 (33%) | 2047/20000 = 10.2% | **4/2000 = 0.20%**（S3 300×200 时 2/2000 = 0.10%） | 同上 |
| Mitten200 X | 39/39 (100%)，160–203 s/shot | 220/240 = 92% | **28/60 = 47%** | S1 50%，S2/S3 ~2%，兜底 47%；130–143 s/shot（兜底 BPOSD 占 60% 时间） |
| Mitten200 Z | 39/39 (100%) | 206/240 = 86% | **30/60 = 50%** | 同上 |

plpr：BB144 3.5e-5 / 1.4e-5（BPOSD 1.7e-4 / 7.5e-4）；Mitten200 1.3e-3 / 1.4e-3（BPOSD 同基 5e-3、两基 ~1）。论文 telescoping 的 Mitten200 在 0.5% 约 1.5e-3 每轮块错 ≈ 12 轮块错 2%。

**结构性结论**：两个码里 BP 各级收敛的 shot **零错**（BB144 S1 3/3850 错为例外，S2/S3 零错）；全部失败来自"BP 不收敛 → 兜底 BPOSD"。BB144 走到兜底的 shot 0.1–0.4%，Mitten200 47–50%。所以 Mitten200 的天花板就是兜底级：论文在这一级用 Gurobi 整数规划 MLE（1 CPU 小时/shot，超时按错计）。下一步：(1) S4 换成整数规划 MLE（HiGHS，先在 GARI 答案侧子问题上做，限时），(2) 按 Relay-BP README 对每个码用固定预算自动调 gamma 区间（Mitten 图与 gross 图不同），(3) S3 预算随 DEM 规模缩放，(4) 兜底 BPOSD 对大 DEM 太慢（160 s）且必错，可改为"计为失败"节省时间。

**成本**：BB144 每 shot 0.6–0.8 s（20k shots/基在 NSCC 16 shard 下约 15 分钟）；Mitten200 每 shot 110–143 s，其中兜底 BPOSD 占 60%，去掉后约 50 s，仍需 watchdog 预算随 DEM 规模缩放。数据：`gari_bb144_p005.jsonl`（S3 100×100）、`gari_bb144_p005_s3x300.jsonl`、`gari_m200_p005.jsonl`。

### 9.3 定案（2026-09-03 晚）：采用论文自己的 decoder —— `telescoping_cpu`（`telescoping_structural_v3_6`）

用户定案："decoder 就换成 telescoping 的源码，其他不动或按这个 decoder 适配"。§9.2 的自研级联 `relay_cascade_gari_v1` 与其协议 id `cascade_structural_v3_6` 同日退役（代码删除，从未上机）；正式 decoder 改为 Mitten 论文开源的 telescoping decoder（github.com/a7b/yarn，MIT，vendored 在 `third_party/telescoping-decoder`，commit `c97099b759dc…`）的 **CPU 配置**：S1/S2（GPU）关、S3-A/B/C（CPU relay 级，包自带 variant 表）在两基 DEM 的 GARI 系统上跑、S4（Gurobi）关，无级接受的 shot（NC）计为逻辑错误。同一两基电路、p=0.5%、T=12、两基 detector 实测：

| 码 | 每基块错（telescoping_cpu） | CPU-s/shot | 对照 |
|---|---|---|---|
| BB144 | X 1/2000、Z 0/2000（扩展 13/24000） | ≈0.07 | 冻结 BPOSD 2.40%/10.2%；自研级联 0.50%/0.20% |
| Mitten200 | X 1/60、Z 1/60 | 11–18 | 冻结 BPOSD 92%/100%（两基 DEM 饱和）；自研级联 47%/50%（130–143 s/shot） |

结论：§0 与 §3 的判断成立——p 固定时杠杆在 decoder。论文 decoder 的 CPU 配置在同一 p=0.5% 两基协议下把 Mitten200 从"饱和"降到百分之一量级、BB144 降到 1e-3 以下，且比自研级联快一个数量级（Mitten200 130–143 s → 11–18 s/shot）。最坏一枪按迭代数封顶（成本模型 `TELESCOPING_COST_S_PER_COLUMN_ITER` × 1.3 × 列数 × 104,000），没有逐级时限，shots 按预算点策略（≤288：20k/200k；≥300：5k/50k）。§4 的 B2（锚点改 0.2%）与 §6 的 `bposd_structural_v3_6` 建议由此作废；机制修复清单（B3–B5、M1–M5）不变，定稿见 06 稿 §1 与 §7 第 6 条。

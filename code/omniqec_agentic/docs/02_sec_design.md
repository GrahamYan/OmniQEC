# OmniQEC Fig.2 重构设计定稿 02：Syndrome-Extraction Circuit 与噪声 Profile 架构

- 状态：**定稿 v1.1**（2026-08-15）
- 依据：《OmniQEC New Fig.2 Experiment：Code Family 与 Slow-Loop 重构规划》PDF §2 + 设计稿 01（定稿）+ 讨论决策 + **两轮文献核查**（SBB 全文逐条核对、Manes–Claes 原文、Beam/LSD 论文与实现）
- 修订记录：v1.0 草案 → v1.1 定稿：并入文献核查的 4 处修正（§2.1 有效性论证归属、§5 residual 集与 extended code 精确定义、§1 引用基线更正、§4.5 Manes–Claes 作用域收窄）、实验模型升级显式化（§2.0）、实现接缝方案（§7.1）、decoder 选型定案（§11，原 03 稿范围提前）、license 核查结论（§9）。语义定义统一沿设计稿 01（命名、config 风格、版本化、metadata 惯例）。
- 关键文献（已核实）：
  - Strikis–Browne–Beverland, "High-performance syndrome extraction circuits for quantum codes", arXiv:2603.05481（下称 **SBB**；本地副本存 scratchpad）
  - Tremblay–Delfosse–Beverland, PRL 129, 050504 (2022), arXiv:2109.14609（coloration + cardinal circuit，同一篇论文）
  - Manes–Claes, "Distance-preserving stabilizer measurements in hypergraph product codes", Quantum 9, 1618 (2025), arXiv:2308.15520
  - Ye–Wecker–Delfosse (IonQ), "Beam search decoder for quantum LDPC codes", arXiv:2512.07057, PRX Quantum (2026)
  - Hillmann–Berent–Quintavalle–Eisert–Wille–Roffe, "Localized statistics decoding", Nat Commun 16, 8214 (2025), arXiv:2406.18655
- 范围：`sec/` 模块（合成、排名、审计）、噪声 profile 架构（4 平台族）、decoder 选型定案；qualification 协议与 slow loop 编排属设计稿 03。
- 替代关系：全量替代现有 `verification/circuit.py` 的调度与发射职责；`verification/noise.py` 版本化 profile 体系保留并扩展；`verification/distance.py` 新增 `d_circ_ext` 通道。

---

## 0. 决策记录

| # | 决策 | 来源 |
|---|---|---|
| D1 | canonical SEC = **left–right circuit（LRC）**：非交错有效性（构造保证）+ 交错级深度 | 已确认 |
| D2 | 合成策略 = **排名式选择**（色置换枚举 + Δmin/τA/W 三段排名） | SBB 框架，已确认 |
| D3 | 审计分级：**仅 static validity 为 hard test**；其余全部实现但不 gating | 用户决策 |
| D4 | 噪声 = **4 平台族**：uniform / si1000 / helios / atom（预留） | 用户决策 |
| D5 | decoder 详细 qualification 协议归设计稿 03 | 用户决策 |
| D6 | **decoder 选型定案**：默认 = Beam（B=64，论文 beam64 超参），保底 = BP+LSD（论文配置）；BPOSD/RelayBP 退出默认集 | 用户决策 2026-08-15，§11 |

## 1. 定位与原则

1. SEC 首先追求跨 family 稳定、公平、确定性与构造级正确性；LER 的**准确率**优先于文献最低 LER。
2. 有效性靠构造保证（非交错性），不靠事后修复循环——修复回退频率若按 family 不均匀会引入系统性不公平。
3. 所有候选使用同一合成 policy：同一枚举序、同一排名规则、同一预算。policy 版本化，跨版本 LER 不可比。
4. AlphaSyndrome / ASC / PropHunt 仅作 finalist sanity comparison。**（v1.1 更正）**它们是**距离优先**（非"深度最优"）的搜索型调度器：SBB 实测其电路深度为 LRC 的 >2×（AlphaSyndrome，[[61,1,9]] 色码）与 >3×（PropHunt，[[108,18,4]] 随机 quantum Tanner 码），真实错误率区间逻辑性能反而更差——不进主 loop 有实测依据。

## 2. Canonical SEC：Left–Right Circuit

### 2.0 实验模型升级（v1.1 显式化）

现有 `verification/circuit.py` 的 memory 实验是 **ver1 单边模型**：X-memory 只调度 HX、Z-memory 只调度 HZ，两类 check 从不同轮交错（这也是现状从未遭遇 X/Z 排序有效性问题的原因）。LRC 是**每轮同时测量两类 check 的完整提取电路**——这是实验模型升级，不只是换调度。连锁后果：detector 数约翻倍、DEM 更密、每轮 idle 结构不同、decoder 吞吐基线需按新电路重测（§11.3）。合法性由 `sec_policy_id` 版本化保证（跨版本 LER 不可比）。

### 2.1 结构定义

**Partition 复用设计稿 01 §7 的双 sector 边界元数据**（bb：两个 ℓm block；hgp：n1·n2 | m1·m2；LP：(nA·nB)|G| | (mA·mB)|G|）。SBB 已证有效性对**任何 CSS 码、任何 partition 无条件成立**（核查确认），四 family 的天然两块结构使我们可跳过 SBB 的 partition 搜索阶段——省时间、保确定性与公平。

记 HX=[LX|RX]、HZ=[LZ|RZ]，对四个子图各做一次边着色（SBB 不要求最小着色；我们**取最小着色**以达最低深度，König 定理下色数 = Δ）。每轮两个窗口：

- 窗口 1：LX 与 RZ 的 2Q 门并行执行，t1 = max(|C(LX)|, |C(RZ)|) 层；
- 窗口 2：LZ 与 RX 并行执行，t2 = max(|C(LZ)|, |C(RX)|) 层。

**Staggering 是有效性机制，不可省**：X-ancilla 在第 1 步制备、t1+t2+2 步测量；Z-ancilla 在 t1+3 步制备、下一轮的 t1+2 步测量（Z 测量周期跨 m 轮窗口 2 与 m+1 轮窗口 1，SBB Fig. 3）。论文的有效性论证：staggering 使同轮内**所有 Z-check CNOT 先于所有 X-check CNOT**（非交错电路），故 detector 确定。（v1.1 归属更正：草案中"同步变体因 X/Z 交错宇称为奇而无效"是**本设计自有推导**，SBB 未做该分析；保留作直觉解释，正式依据用论文的非交错表述。）首轮/末轮存在 partial cycle，boundary 处理单列实现小节。

每 check 一个专属 ancilla、逐轮复用（预算公式 N_data + rows(HX) + rows(HZ) 语义不变）。memory basis（X/Z 双测）、observable 定义、轮数默认 max(3, min(d_hint, 8)) 沿用现约定；detector 布线按 staggered 结构重写。摊销深度 t = t1 + t2 + 2（SBB Eq. 6，深度为 m→∞ 的摊销极限）。

### 2.2 深度预期

| Family | 公式 | w_max=9 预期 | 备注 |
|---|---|---|---|
| `bb` | wA + wB + 2 | **11** | w=6 时为 8，与 Bravyi 电路同深（回归锚点）；饱和 SBB Eq. 3 下界 |
| `hgp` | δ(H1) + δ(H2) + 2 | **≤10** | seed 度 [2,4]；SBB Prop. 1 |
| `lp_ab` / `lp_nab` | δ(A) + δ(B) + 2 | 典型 11–14 | δ 取行/列 cell 权和最大值 |

（v1.1 精确化）`t = δ(A)+δ(B)+2` 恒成立；当 `r(H_XZ) ≥ c(H_XZ)` 时饱和下界 `t ≥ max(c(H_XZ), r(H_XZ)+2)`（SBB Eq. 3；BB w=6 满足）。SBB Appendix A 的深度证明对一般群代数 F₂[G] 成立，唯一前提是正则表示行/列权相等——**对非阿贝尔群成立（本设计推论；论文未显式写 non-abelian）**。lp 的深度驱动项可能是数据列度（01 稿开放 2×2 的物理后果）；max qubit degree 照常记录。

补充（SBB Prop. 4/5，gross 码）：非交错单 ancilla SEC 的 d_circ ≤ 11 < d=12——LRC 的电路距离天花板略低于码距是该结构的已知代价，`d_circ_ext` 通道（§5）会如实测出。

### 2.3 着色算法与确定性

- 二部图最小边着色（König：色数 = Δ），交替路径增广实现；n≈300 规模毫秒级；
- canonical 路径**零随机数**：qubit 索引 canonical（builder 的 sector 序）、边按 (check_idx, data_idx) 字典序、增广路径选择确定；
- （v1.1 增补）可选 SBB 的 linear-sum-assignment 着色变体（把低色号聚集以进一步压 τA）作为后续消融项，v1 用纯 König。

## 3. 合成策略：排名式选择（D2）

### 3.1 候选生成

对每个子图取 canonical 最小着色，再对色类做**确定性置换采样**生成 P 个同深度候选（默认 P=64；置换空间不足时取全部）。（v1.1 修正实现方式）**不用字典序前 P 个**——那会挤在置换空间一角；用固定种子的确定性计数器（splitmix 型）驱动各子图色类置换，零外部随机性、完全可复现、覆盖分散。"零随机数"约束限定于 canonical 路径；候选 i 由计数器序号唯一确定。（对照：SBB Appendix G 用最多 1000 个采样序列做 profile 研究；P=64 是本框架的预算选择，进协议参数。）

### 3.2 排名指标（三段淘汰，v1.1 成本结构化）

指标沿 SBB：Δmin（residual 最小距离，大者优）→ τA（ancilla 空转数，小者优；制备延迟/测量提前内建于 builder，SBB Fig. 7）→ W（residual distance profile，字典序小者优）→ 枚举序号。**实现为三段淘汰以控成本**：

1. **GF2 精确淘汰 Δ=1**（全体 P 个候选）：residual e 是否 ∈ ker(HX)\span(HZ) 的秩检验，零成本；
2. 存活者按 **τA、W** 排序（纯组合量）；
3. 仅头部平手圈用 **BPOSD 距离探测**分辨 Δ≥2 的取值，估计带 cap（只需分辨排序）。探测参数默认对齐 SBB Appendix D（T=100 trials、200 BP iters），可由 config 下调（记录为协议参数）。

可选第 4 档 d_ext 全量复排：默认**关**（SBB 实测该复排收益极小且未用于其最终数值）。

### 3.3 灾难处理

top-1 候选 Δmin=1 → `schedule_catastrophic`：候选隔离、不测 LER、不进父代池；独立于 `degraded_isd`/`degraded_ext` 的电路级信号，反馈给 LLM。同 family 内 canonical 高频灾难 → 工程报警。

（v1.1 作用域收窄，Manes–Claes 核查）**性质测试仅限 `hgp`**：Theorem 2（原文）："任意 stabilizer 测量顺序下 HGP 码的 effective distance 仍为 d"——适用条件为单裸 ancilla per check、无 deformation、d ≥ 2。论文**明确不覆盖 LP**（"for general lifted product codes, no such proof is possible"；LP ⊃ GB ⊃ 非距离鲁棒的 rotated toric）——`bb/lp_abelian/lp_nonabelian` 是调度致距离退化的真实地带，出现 `schedule_catastrophic` 是**合法信号而非 bug**；这正是排名式选择对这三个 family 的价值所在。hgp 触发该 flag 时归因为"管道任意处 bug"（builder、CSS 合法性、detector 标注均可能），非特指调度器；sector_h 开启时定理不适用，性质测试跳过。

### 3.4 协议参数

P、探测 trials/iters、排名规则版本、d_ext 复排开关 → run config + 候选 metadata + slow_results 行。与 shot shards 同等的**测量协议**地位。

## 4. 审计体系（D3：分级已定）

### 4.1 Static validity —— hard test（唯一 gating）

解析**生成出的 Stim 电路本体**（审计工件，捕获 builder 自身 bug）：

- 边集合 = Tanner 边 × 每测量周期恰一次（staggered 下按周期配对计数，不按墙钟轮）；
- 逐 TICK 冲突检查：层内每物理比特至多参与一个 2Q 门；
- ancilla 制备/测量基与 check 类型一致（含 lowering 后的等价形式）；
- 2Q 门方向/orientation 正确（profile lowering 后按门型表复核）；
- HX·HZ^T = 0 工件级复核。

### 4.2 Noiseless Stim validity —— soft，de facto 阻断

无噪声 DEM 构建成功、无 gauge detector；noiseless detector + observable 采样全零；detector 计数闭式断言（staggered 计数公式重新推导，R=2/R=3 双档防 off-by-one）。失败语义 = `synth_error`：管道故障 → 工程报警 + 候选置"未测量"；不作为 degraded 信号、不进排名。

### 4.3 Single-fault DEM 扫描 —— soft，工件级

结构饱和审计噪声模型（全位置、全 Pauli 分量非零，均匀小 p；独立于评估 profile，避免 Helios 零分量审计盲区）→ 无分解 DEM → 线性扫描全部 error mechanism：detector 集为空 ∧ observable 翻转 → 记录。与 §3.2 的 Δmin GF2 精确路径交叉验证，不一致 = builder/lowering bug 直接指示器。

### 4.4 Schedule sensitivity —— soft，仅 per-family top-K

变体 = 排名分布抽样（top-1 + P10/P50/P90，K=4–6）；少量 shots 测 per_logical_per_round；点估计比 >2 且 Wilson CI 分离 → `schedule_sensitive`。ranking 一律用 top-1 的数字，标志与 spread 进产物。hgp 的 spread 应只来自 Nfail/idle 而非距离（Manes–Claes），违背即报警。shots 预算与 adaptive sampling 合并设计（03 稿）。

## 5. 距离证据：新增 `d_circ_ext` 通道

（v1.1 精确化，SBB Thm 1 核查）对非交错电路，circuit distance **精确等于** extended code 距离 d_ext(E)，其中：

- **E 恰为每 check 的 CNOT 后缀集** `{Σ_{s=l}^{w} e_{i_s} : l = 2..w}`（weight-w check 有 w−1 个 residual；l=1 全支撑后缀 = stabilizer 本身，排除）。ancilla idle 错误映射到同一后缀集，**不是独立分量**；
- extended code 同时给 **H 和 logical 矩阵 L** 追加列（每 residual 一列 = 支撑列的 mod-2 和）；d_ext = min{|x| : H^ext x=0, L^ext x≠0}；
- 等式 `d'_circ = d_ext(E)` 仅对**完整** residual 集成立，子集只给上界；两侧组合 d_ext = min(d_ext^X, d_ext^Z)；
- 保持 check 内相对序的交错化距离只增不减（SBB Eq. 12）——未来交错升级路径的安全垫。

实现与语义：

- extended-H 上的 ISD/BPOSD 探测（规模远小于 DEM 核空间），注册进 `verification/distance.py` 为 `d_circ_ext`，**按 X/Z 两侧分别记录**（对齐 d_x/d_z 惯例），性质为"探测所得上界，收敛即等式"，记录 trial 数与 provenance；
- 新增 `degraded_ext` 标志，语义对齐 `degraded_isd`（父代隔离、d_eff/Q_eff 降级）；
- `verified_circuit_distance = min(各通道上界)`（d_circ_ext、DEM-ISD、可选 exact stim）；d_eff = min(d_proxy, verified) 与 d_cap 语义不变；
- **DEM-ISD 保留**为工件级交叉验证（作用于真实电路 DEM，能捕获 lowering/布线 bug；与 d_circ_ext 显著不一致 → 报警）；LER slope 通道不变。

## 6. 噪声 Profile 架构（D4：4 平台族）

### 6.1 平台族与版本

```text
uniform   基线        uniform_paper_v1（v3.5 默认：论文 SI C，无 idle，见 §9.3）/ uniform_legacy_v1（v3.4 及更早历史、golden、saturate_noise）
si1000    超导        si1000_modern_v1 / si1000_measurement_x2_v2（不动）
helios    离子阱      helios_calibration_informed_pauli_v1 / _symmetrized_v1
                      / _pauli_sharpened_gamma2_v2（不动）
atom      中性原子    【预留】命名空间 atom_*_v1；本稿只定接口契约
```

注：profile 参数不动 ≠ LER 数值可比——SEC 从单边模型换为 LRC 后所有绝对值都会移动，由 `sec_policy_id` 版本化标记（§8）。

### 6.2 Profile 注册契约（noise.py 扩展）

每个 profile 声明：①平台族 + 版本串；②native 2Q 门与 lowering 表 id；③各操作类（2Q/1Q/idle-per-tick/reset/measurement）的通道放置规则与参数化；④横轴参数（名称、默认值、CLI 旗标）；⑤slope_offset（非 uniform 一律 None）；⑥provenance。stim_builder 只消费 ②+③；schedule 层完全 profile 无关。

### 6.3 Lowering 表

| 平台族 | 2Q 门 | 说明 |
|---|---|---|
| uniform | CX | 现约定 |
| si1000 | CZ + H 夹层 | H 层计入深度与 idle 记账 |
| helios | RZZ Pauli-vector | orientation 由 check_type + 边方向自然携带 |
| atom | 预期 CZ-native | 若 §6.5-Q2 确认，可复用 si1000 CZ lowering，仅参数不同 |

TICK 严格按层；idle 逐层结算；staggered 制备/测量位置上的 measurement/reset 因子由 lowering 负责放置。LRC 深度低于全着色方案，在 idle 显式计 tick 的三族下是直接 LER 收益。

### 6.4 sector_h 钩子

门型按 (check_type, qubit_sector) 查表：sector-2 上 CX(a_x→q)→CZ、CX(q→a_z)→XCX。着色、宇称结构与 LRC 有效性在局部 H 共轭下不变。v1 默认关（01 稿 §7），开启时 stim_builder 零改动；开启时 hgp 性质测试跳过（§3.3）。

### 6.5 atom 预留位：待决设计问题（不阻塞本稿实施）

1. **Pauli-only vs erasure/loss（关键架构分叉）**：heralded erasure 牵动 Stim 指令、DEM 语义、decoder 动态先验，属管道级变更。**v1 契约钉定为电路级 Pauli 近似**（provenance 记录简化）；erasure-aware 另立设计稿。
2. native 门与 mid-circuit readout 假设（Rydberg CZ 前提；成像测量时长 → idle 惩罚因子）。
3. all-to-all 抽象维持 v1；movement 噪声留 v2 横轴候选。
4. 校准来源与横轴形态（scale 型 vs base-p 型）。
5. `--hardware atom` 与平台族默认映射进 config，profile 落地前显式标注"评估协议未分叉"。

## 7. 模块布局与重构边界

```text
sec/
├── graph.py           # sector 元数据 + HX/HZ → LX/RX/LZ/RZ 四子图（canonical 索引）
├── coloring.py        # 确定性最小边着色 + 计数器驱动的色置换采样
├── schedule_rank.py   # residual 后缀集导出、Δ 三段淘汰、τA、W、排名
├── schedule.py        # LRCSchedule：窗口/层/边/check_type/staggering + policy 版本串
├── stim_builder.py    # LRCSchedule × profile lowering × memory basis → stim.Circuit
│                      #（detector/observable 发射、首尾 partial cycle）
└── audits.py          # static(hard) / noiseless(soft) / single-fault(soft)
```

### 7.1 与现有管道的接缝（v1.1 定案）

- **入口签名保留**：`build_memory_experiment_pair` 的 5 个调用方（slow.py、cli、postsearch、sinter_sweep、verification/__init__）不动；`sec/stim_builder.py` 实现新逻辑，`verification/circuit.py` 变薄转发（slow.py 的 2480 行重构属 03 稿，02 实施不进入）；
- **Schedule 按 code_hash 缓存**：合成 + 排名 + 审计每个候选码只跑一次（schedule profile 无关），stim_builder 按 (profile, basis, noise point) 从缓存 lowering；
- sensitivity audit 的编排逻辑放 slow loop（03 稿接口）；`verification/noise.py` 保留并按 §6.2 扩展。

## 8. 版本化与可比性

- `sec_policy_id = "lrc_ranked_v1"` → run config、候选 metadata、slow_results 每行；跨 SEC 版本 LER 不可比；
- schedule 不进 code_hash（hash 是码的身份）；选中的置换序号、Δmin、τA、W 截断、审计结果进候选 metadata；
- BB72/144 resident baseline 走同一 LRC 流程；"beats BB"= 同 SEC 同 decoder 下击败。w=6 时 LRC 深度与 Bravyi 一致（8），但 schedule 排名与噪声模型不同，绝对值不对齐文献，论文显式声明。

## 9. 验证与回归（v1.1 含 license 结论）

**License 核查结论**：SBB 代码仓（github.com/PurePhys/LR-circuits）**无 LICENSE 文件**（默认保留所有权利）——**不得 vendor 其代码，`sec/` 按论文构造自研**；Zenodo 参考电路数据集（DOI 10.5281/zenodo.18853600）为 **CC-BY-4.0**，黄金样本合法可用。

回归项（**测试先行**：黄金样本对拍在 stim_builder 实现之前就位）：

1. SBB Zenodo 参考电路结构级 diff + DEM 等价性抽查；
2. `bb` 深度断言：w=6 → 8（Bravyi 锚点）、w=9 → 11；
3. staggered detector 计数公式 R=2 / R=3 双档；
4. `hgp` 距离免疫 property test（作用域按 §3.3：无 deformation、d≥2、单裸 ancilla）；
5. Δmin GF2 精确路径与 single-fault DEM 扫描一致性；
6. sector_h 开启下全审计套件复跑；
7. Mitten [[150,30,10]] 对读扩展：LRC 深度按 §2.2 公式断言。

### 9.1 实现笔记（2026-08-15，黄金对拍后钉定的三个协议细节）

黄金对拍（HGP625 双 basis，84 段逐段语义零差异，含全部噪声指令）过程中从参考实现钉定了三条 §2/§4 未及明确的协议细节，已实现于 `sec/stim_builder.py`：

1. **Detector 单 basis 约定**：memory 实验测量两类 check，但只对同 basis check 发射 detector（X-memory 仅 X 检测器）；计数闭式 = `(R+1)×m_basis`。§2.0 预估的"detector 翻倍"不发生，DEM 密度增幅相应更温和。
2. **末轮截断 + 推迟测量**：最后一轮不启动新 Z 周期（无复位、无 LZ 层）；悬挂的 Z 测量推迟到终段，顺序为 M(Z-anc, 有噪) → MX(X-anc, 有噪)+末轮检测器 → DEPOLARIZE1(data) → 无噪声数据读出 → 重建检测器 → observables。
3. **Idle 噪声 = SBB Fig.7 压缩的选择性省略实现**：ancilla 仅在本 reset→measure 周期的首末 CNOT **之间** idle 时受噪（制备延迟/测量提前）；数据比特按 sector 组活跃窗口受噪（左组自 prologue 起、右组自窗口 1 起，末轮各自退休）；终段数据额外获得恰一次 DEPOLARIZE1。

黄金样本（CC-BY-4.0）已 gzip 固化于 `tests/golden/`，回归 `tests/test_sec_golden.py` 含双 basis 全段对拍 + 深度锚点（bb w6→8 / w9→11）+ 四 family noiseless 有效性。

### 9.2 实现笔记（2026-08-15，§6.3 lowering 落地 + 采样源协议变更）

4. **si1000 lowering（CZ + 窗口级 H frame）**：CX(c→t)=H(t)·CZ·H(t)，相邻 H·H 相消后 frame 提升到窗口粒度——每个 ancilla 在（制备延迟的）reset 后、（测量提前的）measurement 前各得一层 H；数据比特仅在自身 X-check 窗口持有 H frame（左 sector=窗口 1，右 sector=窗口 2）。具体：全局 prep 后一层（zanc，X-basis 时加 data 的 |+⟩ 制备），每轮四层——step 2 前 {左 data, xanc}、step t1+2 前 {全 data, zanc}、step t1+3 后 {zanc}、step t 前 {右 data, xanc}。每层 H 是完整 tick，计入深度与 idle 记账（实测 BB72 R=4：34→50 tick）。通道率沿用旧实现：CZ→DEPOLARIZE2(p)、H→DEPOLARIZE1(p/10)、R→X_ERROR(2p)、M→X_ERROR(factor·p)；ordinary idle p/10 沿用 uniform 的 compaction 门控；resonator idle 2p 施于含 R/M 的 tick 上除 R/M 靶外的全部 compaction 活跃比特。终段：数据在 ancilla 读出 tick 获 p/10+2p，X-basis 读出经有噪 H，X_ERROR(factor·p) 后无噪声读出。
5. **helios lowering（RZZ Pauli-vector）**：保持 CX 调度，每个 CX 层展开为三个原生矩：SQRT_Y_DAG(targets) | SQRT_ZZ(pairs)+虚拟 S_DAG(controls) | SQRT_X_DAG(targets)+虚拟 S_DAG(targets)。RZZ 噪声 = PAULI_CHANNEL_2，Pauli 向量按 (control, target) 序——orientation 由 check 类型+边方向自然携带（X-check: xanc 先；Z-check: data 先）。X-ancilla |+⟩ 制备/读出降为 SQRT_Y_DAG+虚拟 Z，边界位置与 si1000 H 层相同。1Q 旋转 DEPOLARIZE1(scale·p1)、R→X_ERROR(scale·p_init)、M→X_ERROR(scale·p_meas)、虚拟门无噪、无 idle 噪声（校准模型无此项）。DEM 需 `approximate_disjoint_errors`（阈值即 `HeliosNoiseConfig.approximate_disjoint_errors`，分量上界已在 config 校验）。
6. **static audit 的 profile 分工**：static 审计解析 uniform 的 CX/RX/MX 结构；非 uniform profile 下 seam 对**同一 schedule** 额外发射一份 uniform 电路做审计（schedule 层 profile 无关，02 §6.2），实验电路用 lowering 发射。验证：零噪声行列式性（任何 frame/共轭错误都会使 detector 随机触发）× 6 profile × 2 码（D6 [[24,8,3]]、BB72）× 2 basis 全过，detector 计数 (R+1)·m_basis 不变。
7. **采样源协议变更（sampling_source = noisy_circuit）**：slow loop 的 LER 采样从 DEM 独立机制采样切换为含噪电路直接采样（`compile_detector_sampler` + `separate_observables`）。动因：DEM 独立机制近似在 hook 富集码上系统性低估 LER（D6 实测差 ~20%，p=0.005 时 0.0645 vs 0.0781）。decoder 仍消费未分解 DEM 矩阵；仅采样侧换源。Sinter 迁移门结论（passed-with-attribution）：共享电路/DEM/采样协议后，与 ldpc 栈的残差全部归因于其 DEM→矩阵转换的列去重（468 vs 我们忠实转换的 525 列），同 shot 对比我们的栈更优（17.2% vs 22.1%）。

### 9.3 实现笔记（2026-09-02，噪声 profile 回归论文 SI C；协议 `bposd_structural_v3_5`）

用户决定：「SEC 这么做没有问题（LRC 保留），但是引入的 noise 这块还是按照我组员的论文里面来，这个 idle 噪声不要。」SEC 保持 LRC（§2、§9.1 的调度 / detector / 终段约定全部不动），但**正式协议的噪声模型改回组员论文 SI C b**。原文：

> All circuits use a uniform circuit-level depolarizing model controlled by a single physical error rate p. Each qubit reset is followed by single-qubit depolarizing noise of strength p; each controlled-NOT is followed by two-qubit depolarizing noise of strength p; each ancilla measurement is preceded by a bit-flip error of probability p; and the final data readout is preceded by a basis-conjugate flip of probability p.

8. **新 profile `uniform_paper_v1`**（`verification/noise.py` 的 `NoiseProfile.UNIFORM_PAPER`；`sec/stim_builder.py::build_lrc_memory_circuit` 的 `kind == "uniform_paper"`）：与 legacy uniform 共用同一份 LRC 分段 / 调度 / detector / observable 发射，**只有噪声指令不同**：
- 每个 reset（全局数据制备 `R`/`RX` **以及**每个 ancilla 的 `R`/`RX`）：先发 reset，再对同一批目标发 `DEPOLARIZE1(p)`；
- 每一层 `CX`：`CX` 后紧跟 `DEPOLARIZE2(p)`；
- 每个 ancilla 测量：`M` 前紧跟 `X_ERROR(p)`，`MX` 前紧跟 `Z_ERROR(p)`（与 legacy 相同）；
- 终段：推迟的 `M(zanc)` / `MX(xanc)` 按上一条加噪；之后**不再**对数据发 `DEPOLARIZE1`；终局数据读出前发基共轭翻转——X-memory 为 `MX(data)` 前 `Z_ERROR(p)`，Z-memory 为 `M(data)` 前 `X_ERROR(p)`；
- **任何位置都没有 idle `DEPOLARIZE1`**（phase-2 的 idle 标注对该 kind 不发射）；TICK 结构不变；
- `saturate_noise=True`（结构审计）仍走 legacy uniform lowering，与非 legacy profile 组合仍 `ValueError`；`noise_point=None` 仍表示 legacy 发射（golden 对拍、审计路径逐字节不变）。
   detector / observable / 测量计数与同一 schedule 的 legacy 发射逐项相等（回归 `tests/test_v35_noise_profile.py`）。
9. **§9.1 第 3 条的 idle 规则只保留给 `uniform_legacy_v1`**（历史：v3.4 及更早的产物；`tests/golden/` 对拍；`saturate_noise=True` 的结构审计）。`noise_point=None` 仍表示 legacy 发射，golden 回归逐字节不变。§6.1 的平台族表按此重读：uniform 族现有两个版本——`uniform_paper_v1` 是正式默认（`NoiseSweepConfig()` / `uniform()` / CLI `--noise-profile` 默认 / `--p` 别名），`uniform_legacy_v1` 经 `uniform_legacy()` 显式选用；si1000 / helios 的 lowering（§9.2 第 4–5 条）不受影响。§6.3 末句「LRC 深度低于全着色方案，在 idle 显式计 tick 的三族下是直接 LER 收益」对正式协议不再适用（`uniform_paper_v1` 无 idle），对 si1000 仍成立。
10. **为什么**（本机 2026-09-02 诊断）：§9.1 第 3 条是 SBB Fig.7 压缩的忠实实现，但它让噪声体量随 check 权重与 LRC 深度增长、按 family 不均匀——
- 噪声体量（本机 2026-09-02 计数，R=12）：LRC legacy 发射对 Mitten [[200,40,12]]（GAP(40,5)，w=9）产生 **20,248** 个单比特 `DEPOLARIZE1` 目标（idle + 终段数据），论文模型只有 **2,120**（仅 reset 之后）——单比特噪声暴露 **9.6×**，总噪声事件 1.45×；BB144（w=6）只有 1.8× / 1.05×。idle 噪声因此是一个**随 check 权重与 LRC 深度增长、按 family 不均匀**的额外惩罚，系统性压低高权重 LP/非阿贝尔码。
- 测量（本机 BPOSD 50/10，T=12）：Mitten 码在 legacy（LRC + idle）下 p=0.005 两基 LER **= 1.0**（2600/2600），p=0.002 每基 0.78–0.80；同一 LRC 电路去掉全部 idle `DEPOLARIZE1` 后 p=0.002 降到 0.18–0.19，但 p=0.005 仍 0.97–0.98。按论文 SI C 协议（论文自身的**单边顺序** SEC + 无 idle）p=0.005 得 LER_X=0.0019 / LER_Z=0.0038、plpr≈1.2e-5——论文数量级可复现。
- 结论：idle 项是 v3.5 要去掉的、可归因且按 family 偏置的部分；去 idle 后 LRC 与论文单边 SEC 之间在 p=0.005 仍有差距，这是用户明确保留 LRC 的已知代价，不在 v3.5 范围内。LRC 的 hook 结构（Δmin / τA / `d_circ_ext` 排名、SBB Thm 1 的 residual 集）保持不变：Mitten 码 `d_circ_ext(x,z)=(21,16)`、无 `delta_one` 淘汰，不是调度问题。

### 9.4 实现笔记（2026-09-02，SEC 换为 `twobasis_nonint_v1`；LRC 降为历史策略 `lrc_ranked_v1`）

用户决定（同日第二次，原话）：「单一的基肯定是不对的 … 后续 helios 需要做 bias … 必然需要两个基」「不做任何的交错，两个基完全独立开」「基内部可能还是需要 scheduling 来减小 hook 的影响，但开销会小很多」「杜绝一切可能造成大码距或者大 k 被误杀的情况」。证据（本机，BPOSD 50/10，T=12，p=0.5%，两者都无 idle）：Mitten [[200,40,12]] LRC LER_X/Z = 0.98 / 0.97 vs 论文单边顺序 SEC 0.002 / 0.004；BB144 LRC ≈ 0.07 vs 0/1040 shots。去 idle 后差距仍在两个数量级以上，是 §2 LRC 时间线本身的代价。完整记录在 docs/04 §10「2026-09-02 追加（七）」，本节只记与本稿 §2–§5 的对应关系；权威顺序仍是代码 > 04 > 本稿。

11. **§2 的 LRC 不再是正式 SEC**：`sec_policy_id` 由 `lrc_ranked_v1` 改为 **`twobasis_nonint_v1`**（`config.SecPolicyConfig` / `DEFAULT_SEC_POLICY`：`sec_candidates=16`、`sec_cert_iters=400`、`sec_cert_seed=0`；`LRC_SEC_POLICY` = 64 / 100 对应 §3.4 与 §14 第 7 条的旧常量）。`verification/circuit.py::build_memory_experiment_pair` 新增 `sec_policy=` / `layer_blocks=`；`lrc_ranked_v1` 仍可显式选用（golden 对拍走它），strict 拒绝。synth cache key 含策略 id / K / iters / seed / `layer_blocks` 哈希，两条策略永不共享缓存项。
12. **时间线（不交错、两基）**：`sec/twobasis_builder.py`。全局只制备数据（X-memory `RX(data)` / Z-memory `R(data)`）；每轮 `RX(xanc)` → X 层 1..|LX|（`CX(xanc→data)`，每层一个 tick、每层是 matching）→ `MX(xanc)` → `R(zanc)` → Z 层 1..|LZ|（`CX(data→zanc)`）→ `M(zanc)`；终局 X-memory `MX(data)` / Z-memory `M(data)`。每轮 |LX|+|LZ|+4 tick（`TwoBasisSchedule.depth`），`rounds ≥ 1`。§2.1 的左右分区、§2.2 的深度预期、§3.2 的 Δ / τA 三段淘汰只对 `lrc_ranked_v1` 有效。
13. **层**：`sec/layers.py::build_layer_sets`。bb / lp_abelian / lp_nonabelian 携带 `CompiledCode.layer_blocks`（`layer_blocks_v1`：每个环项 = check 块 × 数据块之间的置换 `[cb, db, perm]`，`H_W[cb·g+perm[t], db·g+t]=1`），以块为顶点、项为边建二分多重图，`sec/coloring.py::color_bipartite_multigraph`（边索引版 König，§2.3 的 `minimal_edge_coloring` 字节不变）最小边着色，层数 = max(行权, 列权)（BB = w = 6，D6 lp = 8、C3×C3 lp = 7）；hgp / 无块元数据回退 `minimal_edge_coloring(h)`。两路径都断言每层 matching、边覆盖恰一次。
14. **Detector 两基**：W 基 check 第 1 轮单项、之后相邻配对；反基第 1 轮不发（随机）、第 2 轮起相邻配对；终局只对 W 重构；observable = W 逻辑。计数 = T·m_W + (T−1)·m_opp + m_W。共享发射器 `sec/stim_builder.py::_emit_circuit(segments, _Timeline, _Lowering, ...)` 以 `detect_kinds` / `first_event_kinds` 区分两条时间线；LRC 取 `{W}` / `{W}`，输出逐字节不变（新增改动前快照对拍）。§6.3 的四种 lowering 由同一发射器服务：`uniform_paper` 为正式协议；`uniform_legacy` 的 §9.1 第 3 条 idle 规则仍只用于 golden / `saturate_noise`；`si1000` / `helios` 在新时间线上**只保证正确性**（X-memory 第 1 轮 prep-H + frame-H 不合并、无 window hoisting），不作硬件阶段结果。
15. **基内层序 + 认证**（`sec/twobasis_rank.py::rank_twobasis_schedules`，替代 §3 的三段排名）：每基独立，候选 i=0 canonical、i=1..15 = `counter_permutation(len(L), candidate=i, slot=0(X)/1(Z))`。不交错 ⇒ SBB Thm 1 精确（电路距离 = 扩展码距离），不编译电路：按层序取每个 check 的 residual 后缀（`schedule_rank.residual_suffixes` 语义），`extended_matrices` + `kernel_basis` + `_proxy_side`（ISD 400 iters/side，seed = `sec_cert_seed + i·7 + side`，side 映射同 `d_ext_probe`）→ `d_ext`；`delta_one_residuals` 命中即淘汰。**ISD 估计是上界、只作证据**：`d_ext_x/z`、`chosen_index_x/z`、`per_candidate`、`schedule_degraded_x/z` 全部记录，从不作停止；某基全部候选 delta-one → `ScheduleCatastrophicError`（§3.3 灾难处理不变）。选择 = 每基取最大 `d_ext`、平手取最小序号。返回 `RankedSchedule`（`tau_a=None`、`stage3_group=1`），`evaluation/evidence.py::sec_ext_evidence` 不改读法、只加 `sec_policy_id` 与 `schedule_degraded_x/z`。
16. **静态审计**：`sec/audits.py::audit_twobasis_static`（css 正交、同 tick 无比特冲突、门向：只允许 xanc→data 与 data→zanc 且与 H 一致、prep/meas 基匹配、每条 Tanner 边每轮恰一次、CX tick 数 = rounds·(|LX|+|LZ|)、detector / observable 计数、noiseless DEM 可建）在 `verification/circuit.py` 的 `audit_static` 位置接入；§4.1 的 gating 地位不变。
17. **不声称**：hook 错误未被消除（只被精确认证并在层序间择优）；§5 的 `d_circ_ext` 跨策略不可比；§6.4 sector_h 在两条策略下仍 `NotImplementedError`（§14 第 1 条），`layer_blocks` 不感知 mask；§14 第 7 条的 `LRC_SEAM_*` 常量仍在（须等于 `LRC_SEC_POLICY`），快照改读 `verification.sec_policy`。

## 10. Hook surrogate 处置

Δmin / W / τA 是确定性可计算的 hook 风险特征，覆盖 surrogate 预测目标的大部分。"重训 or 退役"决策移至 03/04 稿；若保留，特征集换成 family 无关的 (Δmin, W 截断, τA, 深度, 列度分布)。（01 稿实施时旧 gb/lp 特征已改写为 bb/lp 新格式，当前可用但弱。）

## 11. Decoder 选型（D6 定案；qualification 协议留 03 稿）

### 11.1 默认 decoder：Beam Search（B=64）

Ye–Wecker–Delfosse (IonQ)，arXiv:2512.07057 / PRX Quantum (2026)；开源 github.com/ionq-publications/BeamSearchDecoder（C++ 核 + Cython/Python，**CC BY-NC-SA 4.0 非商用**——学术使用可，商用需谈判，选型知悉）。

论文只基准 B ∈ {8, 32, 64}；B=64 是最强档。**必须配套论文 beam64 超参**（repo 构造默认是 beam8 调参）：

```python
BeamSearch(model=dem,        # stim DEM 直接消费；保留未分解超边（XYZ 相关性对 beam 有增益）
    beam_width=64,
    max_rounds=20,           # beam8/32 用 10；beam64 用 20
    initial_iters=40,        # beam8 用 30
    iters_per_round=30,      # beam8 用 20
    num_results=1)           # 32 = 列表解码（LER 再降至 17×），runtime 论文未报——config 开关，qualification 实测后定
```

依据（[[144,12,12]]，电路级 p=1e-3，单核 M3 Pro）：LER 比 BP+OSD-CS-10 低 **7.0×**（num_results=32 时 17×）；平均 6.7ms/shot（BPOSD 10.6ms），99.9 分位 17.7ms（BPOSD 289ms）。HGP [[450,32,8]] 上收益 3.7×——增益随码而变，qualification（03 稿）按本框架 LRC 电路实测。无内部 shot 并行（由 slow 进程池负责），自带 sinter adapter。

### 11.2 保底 decoder：BP+LSD（论文配置）

Hillmann et al. 论文全部电路级实验：min-sum、30 iters、α=0.625、parallel、LSD-0（已匹敌 BP+OSD-CS-7）。本框架保底配置（`LsdDecoderConfig`，2026-08-15 已修复落地）：

```python
max_iter=50, bp_method="ms", ms_scaling_factor=0.625,
lsd_method="lsd_cs",   # ldpc 2.4.1 陷阱：不显式传则 lsd_order 被静默清零
lsd_order=8            # 论文：高阶增益小但一致，BB-144 获益最大
```

**历史结果更正**：2026-08-15 前标为 "lsd order8" 的所有结果实际为 BP+LSD-0（α=1.0）——结果有效（LSD-0 即论文头牌配置），标签需重解读。sinter 路径（`SinterLsdDecoder`）在 ldpc 2.4.1 无法表达 lsd_method，已显式钉 order 0 保持标签诚实。

### 11.3 对 03 稿的前置约束

1. LRC 使 DEM 更密（SBB 实测 +8.4%~+14% 密度，BPOSD 解码时间 ×1.3–2，**基线为 unstructured sequential 电路**——v1.1 更正）；qualification 吞吐必须在本框架 LRC 电路上实测，文献与旧单边电路数字均不可用；
2. `decoders` 默认集 = (beam, lsd)；BPOSD 降为 reference 选项、RelayBP 保留为可选（均不在默认集）；
3. adaptive sampling 与 sensitivity audit 的 shots 预算合并设计；
4. qualification set（4 families × 3 sizes × 2–3 instances）在 01+02 落地后生成。

### 11.4 v3.6 定案（2026-09-03）：正式协议 decoder = `telescoping_cpu`（`telescoping_structural_v3_6`）

11.1–11.3 与 v3.2–v3.5 的冻结 BPOSD（50/10）保留为历史；正式协议自 `telescoping_structural_v3_6` 起改为 **Mitten 论文开源的 telescoping decoder 的 CPU 配置**（用户 2026-09-03 定案："decoder 就换成 telescoping 的源码，其他不动或按这个 decoder 适配"；06 稿 §1，`config.TelescopingDecoderConfig` / `TELESCOPING_PROTOCOL_FIELDS`，`verification/decoders.py::TelescopingObservableDecoder` 是 `telescoping_decoder.TelescopingDecoder` 的薄适配层；包 vendored 在 `third_party/telescoping-decoder`，github.com/a7b/yarn，MIT，commit `c97099b759dcb53520d0ce210c6893dd9e0e0ea3` = 协议身份）。同日更早的自研级联 `relay_cascade_gari`（`cascade_structural_v3_6`，`verification/gari.py` + `CascadeObservableDecoder`）已删除、从未上机，其前缀与 `bposd_structural` 一样在 `--resume` 时按名拒绝。

- 输入：两基 DEM + 发射器记录的每 detector 基标签（`MemoryExperiment.detector_kinds`，随 DEM bundle 落盘，worker 不解析电路），包按标签做自己的 GARI 变换（`s3_system="gari"`）；两基的 telescoping 系统在 DEM build 时物化一次（`materialize_telescoping_artifacts`，GARI npz 落在 bundle 旁），decode shard 只加载，并交叉核对发射器的 detector 种类与包的 mask、系统的 detector / observable 计数与矩阵。
- 级：S1/S2（GPU flooding BP / GPU relay-BP）**关**（无 CUDA）→ S3-A（独立 variant 集成，包默认表 5 个，`a_iters=100`）→ S3-B（relay-BP 带记忆 + coset quorum，`b_runs=40 × b_iters=100`，包默认表 6 个先验 variant，`b_quorum=3`）→ S3-C（顺序参数扫描，包默认表 12 个配置，含噪配置各 `c_n_seeds_default=10` seed）→ S4（Gurobi 整数规划认证）**关**（无 license，CPU-only）→ **NC（无级接受）计为逻辑错误**（`nc_policy="fail"`，论文对未解出 shot 的规则；对所有码同一规则，LER 是上界；不再调 BPOSD，没有任何兜底）。S3 的 variant 表用包自带的（`variant_tables="package_default"`，从不改）；全部字段都是协议身份（改任一项须换 profile id），与预算点无关。
- 确定性：包从 `(base_seed, shot id)` 派生每个 S3 seed；适配层以 shard seed 为 `base_seed`、shard 内 shot 序号为 id，解码是 (矩阵, shard seed, shot 序号) 的纯函数，跨机器、pool 宽度与重跑逐 shot 一致（`tests/test_telescoping_decoder.py`）。每 shard 记录 `stage_counts` / `stage_failures` / `stage_time_s` / `declared_failures`（键 `S3A` / `S3B` / `S3C` / `NC`）与包的 per-shot 标签，按基汇总进 slow 证据与 ledger，"因 decoder 放弃而测差"可审计。
- 成本：S3 各级按迭代数封顶，最坏一枪是成本模型而非墙钟——`TELESCOPING_COST_S_PER_COLUMN_ITER`(2e-8 s) × 1.3 × 两基 DEM 列数 × `worst_case_iterations()`（包默认表 104,000 次 BP 迭代）；decode watchdog = max(v3.4 项, shots × 最坏一枪)；`budget_policy` 只含 shots（≤288：20k/200k；≥300：5k/50k），没有逐级时限。
- 依据（本机 2026-09-03，p=0.5%，`uniform_paper_v1`，两基电路，T=12；05 稿 §9.3）：BB144 X 1/2000、Z 0/2000（扩展 13/24000），0.07 CPU-s/shot；Mitten200 每基 1/60，11–18 CPU-s/shot——对比冻结 BPOSD 的 Mitten200 92%/100% 失败、BB144 2.4%/10.2%，以及退役的自研级联的 BB144 0.50%/0.20%、Mitten200 47%/50%（05 稿 §9.2）。
- 依赖：`pip install -e third_party/telescoping-decoder`（editable，launcher 自检核对包旁 `VENDORED_COMMIT`）；两个 C kernel 首次使用时用 gcc 编译进 `$XDG_CACHE_HOME/telescoping_decoder`（launcher 序言 `python -m telescoping_decoder.build` 预编译一次，worker 继承）；relay-bp 与 scipy(HiGHS) 不再是协议依赖。
- 资格测试（`qualification.py`，`docs/qualification/qualification_set_v1.json` v1.1）：参照 = `telescoping_cpu` 默认配置，集合 BB72 / BB144（1000 shots/基）+ Mitten150 / Mitten200（100 shots/基），验收见 06 稿 §1.3。

## 12. 实施顺序

1. ~~groups/grammars~~（01 稿已完成）；
2. **SBB Zenodo 黄金样本获取 + 对拍脚手架**（测试先行，v1.1 提前）；
3. `sec/graph.py` + `coloring.py` + `schedule.py`：canonical LRC + uniform lowering；
4. `audits.py` static（hard）+ noiseless（soft）；
5. `schedule_rank.py`：Δ=1 GF2 精确路径先行，BPOSD 估计路径次之；
6. `d_circ_ext` + `degraded_ext` 接入 distance/ranking；
7. si1000 / helios lowering；
8. Beam decoder 接入 + LSD 配置（LSD 修复已落地）；
9. sensitivity audit 编排（随 03 稿）；
10. atom profile（待 §6.5 决议，纯 lowering 接入）。

## 13. 非阻塞待办

- staggered 首尾轮 detector 公式推导记入实现笔记；
- ~~SBB 包 license 核查~~（已定：代码不可 vendor，Zenodo CC-BY-4.0 可用）；
- d_ext 复排默认关的消融（SBB 实测收益极小，优先级降低）；
- linear-sum-assignment 着色变体消融（§2.3）；
- §6.5 五个 atom 问题决议（用户）；
- num_results=32 的吞吐实测（qualification 时）；
- AlphaSyndrome / ASC finalist 对比协议细则（03/04 稿）。

## 14. 实施修订（2026-08-26，文档漂移校正）

按「代码 > docs/04 > 设计稿」的权威序，记录本次全仓核对确认的 §1–§13 与实现的偏离。正文（含 §9.1/§9.2 两则实现笔记）保持定稿不动，以下每条都给出可复核的代码位置，读者遇到冲突以本节为准。§11 的 decoder 选型本次逐项复核**无偏差**（beam：`beam_width=64`/`max_rounds=20`/`initial_iters=40`/`iters_per_round=30`/`num_results=1`；lsd：`max_iter=50`/`bp_method="ms"`/`ms_scaling_factor=0.625`/`lsd_method="lsd_cs"`/`lsd_order=8`；默认集 `(beam, lsd)`），故不列条目。

1. **§6.4 的 (check_type, qubit_sector) 门型表未实施，「开启时 stim_builder 零改动」不成立**：`verification/circuit.py:155-159` 对任何非平凡 H-mask 直接 `raise NotImplementedError("sector_h under LRC awaits the (check_type, sector) gate table (02 §6.4)")`；`sec/stim_builder.py` 无任何 `h_mask` 代码路径（grep 零命中），该查表尚未存在——落地前 sector_h 出不了电路与 DEM，方向恰与 §6.4 所写相反。作用域要说清：变形并非全线空转，构造侧照常工作——`construction/build.py:57` 解出 mask、`:71` 折进 `code_hash`、`:148-149` 导出 `h_mask_weight`/`h_mask_sha256`（对应 CSV 列 `results_csv.py:24-26`），fast/structural 阶段照常携带（`verification/structural.py:197,201,211-212`），只有电路阶段拒绝。失败形态是**逐候选静默降级**而非崩溃：调用点都把异常收进状态字段——`cli.py:770-772`（`status="error"`）、`evaluation/pipeline.py:199-206`（`circuit_error`）、`evaluation/worker.py:78-79`（`build_error`）与 `:213-216`（`circuit_error`）——所以 `--deformation-mode sector_h`（`cli.py:57-62`，挂在 run-funsearch / run-agentic-cycle / build-dem-smoke / slow-verify-queue 四个子命令）跑完得到的是一整轮全 `circuit_error` 的候选、退出码仍为 0。该开关按 01 稿 §7 仍是合法的 ablation 入口，不宜摘除（`evaluation/common.py` 的 promotion 记录也独立接受该值）；防呆应在参数处理期前置硬失败，而不是让它逐候选沉底。

2. **§4.2/§4.3 的两档软审计已实现但未接进管道**：`audit_noiseless`（`sec/audits.py:100`）与 `audit_single_fault`（`:137`）按稿实现，`run_all_audits`（`:174`）却全仓无调用方（`sec/__init__.py:15` 仅再导出），管道实际只调 `audit_static`（`verification/circuit.py:208`，失败即 raise）。`:190` 的 `synth_error` 键因此从未产生，§4.2 末句「工程报警 + 候选置未测量」的语义尚未接线。但覆盖不是空白，两档要分开看：§4.2 的无噪声确定性与 `(R+1)·m_basis` 计数断言由回归 `tests/test_sec_golden.py:124-144`（四 family × 双 basis）承担（§9.1 末段已记），仍缺 R=2/R=3 双档（§9 回归项 3）；§4.3 的单故障隔离目标由 §3.2 的 GF2 精确路径在管道内达成——`sec/schedule_rank.py:207-222` 淘汰 Δ=1 候选、全灭时置 `catastrophic`，`verification/circuit.py:186-190` 抛 `schedule_catastrophic`，`evaluation/evidence.py:36-38` 把 `schedule_catastrophic`/`delta_one_eliminated` 记进证据，真正缺的只是饱和噪声 DEM 扫描这条工件级交叉验证（§9 回归项 5）。`sec/stim_builder.py` 的 `saturate_noise` 分支目前只为这条未接线的审计服务。补接线不是零成本（`audit_single_fault` 要对每候选双 basis 展平饱和 DEM，`run_all_audits` 还自己重建一次电路），应另行定范围，不并入本次校正。

3. **W 未实现，Δ 的标量只在头部平手圈算出**：三段排名的实际排序键是 stage 2 的 `(τA, 候选序号)`（`sec/schedule_rank.py:223`）与 stage 3 的 `(−d_ext, 候选序号)`（`:231`），两处剩余平手都由枚举序号确定性打破；`W`（residual distance profile）全仓无实现，模块 docstring `:18-19` 已自记 deferred。stage 1 与 §3.2 一致（Δ=1 二值精确淘汰），但对全体 P 个候选只给二值答案（`:207`）；Δ≥2 的标量取值仅 stage 3 算——`d_ext_probe`（`:164`）在 `:228` 对头部平手圈逐个跑，而该圈被 `STAGE3_GROUP_CAP=8` 截断（`:225`，见第 4 条），因此「全体候选的 Δmin」从不存在，落盘的也不叫 Δmin，而是胜者的 `d_circ_ext_x`/`d_circ_ext_z`。连带后果是 §8 所列五项候选 metadata 只落了三项：选中的置换序号以 `candidate_index` 之名进 memory experiment metadata（`sec/schedule.py:99` → `verification/circuit.py:43`，与第 7 条的 `sec_policy_id` 同一条路），τA 与 Δ 以 `tau_a`/`d_circ_ext_x`/`d_circ_ext_z` 之名进 `sec_ext_evidence`（`sec/schedule_rank.py:43-52` → `evaluation/evidence.py:30-40`，同处还有 `delta_one_eliminated`/`stage3_group`/`schedule_catastrophic`）——但 Δ 这一项名窄义也窄，只是胜者的 ISD 上界，不是全体候选的 Δmin。真正没有落点的是「W 截断」与「审计结果」：static 是管道唯一实跑的审计（`verification/circuit.py:208`），失败即 raise、通过则不写任何字段（`MemoryExperiment.to_metadata` `:31-58` 无 audit 键，`sec_ext_evidence` 亦无）。另注意 `chosen_candidate_index` 只活在 `RankedSchedule.metadata`（`:241-247`）里，按第 7 条该 dict 无消费方，不算落点。§0 的 D2 表与 §7 模块速查中 `schedule_rank.py` 的「W」一项按此重读。§10 的特征集本就是条件性设想（「若保留」），hook surrogate 已按 03 D12 退役（`ranking.py:18`），该条不再有实现义务。

4. **stage 3 的探针是自研随机化 ISD 而非 BPOSD；「200 BP iters」无对应参数，「可由 config 下调」未实施**：`sec/schedule_rank.py:33` 引入 `verification.proxy_distance._proxy_side`，`:185` 是该路径上唯一的距离调用；`verification/proxy_distance.py:35-70` 是纯信息集译码（随机列置换 → `_systematic` → 最轻行与两两异或扫描，`t_lightest=24`），无置信传播环节，故 §3.2 的「200 BP iters」在实现中没有对应物。这不算越界——§5（第 133 行）本就写「ISD/BPOSD 探测」，代码取了 ISD 分支；排序量仍是 §3.2 的 Δ（即 §5 的 d_ext，SBB Thm 1），性质为 ISD 上界（收敛即等式），同时充当 top-1 的 `d_circ_ext` 证据。唯一旋钮是 `isd_iters`（`DEFAULT_ISD_ITERS = 100`，`:39`，对齐 SBB Appendix D 的 T=100），但生产路径写死模块常量 `LRC_SEAM_ISD_ITERS = 100`（`verification/circuit.py:87`，`:169` 进合成缓存键、`:181` 进调用）并冻进协议快照 `sec_synthesis.isd_iters`（`workflow/state.py:135-138`）——§3.4 的「协议参数」地位成立，「可由 config 下调」未实施。**易踩的坑**：config 里名字相近的 `dem_isd_iters`（`config.py:228`）是 DEM-ISD 通道的预算（`verification/distance.py:156`），与本探针无关，调它不会改变 stage 3。另有未记于稿的 `STAGE3_GROUP_CAP = 8`（`:40`、`:225`）：§3.2 的「估计带 cap」实现成的是头部平手圈的**组大小**上限，`_proxy_side` 自身跑满 `iters`、无早停。

5. **`d_circ_ext` 的模块地址与本稿不符**：`verification/distance.py` 内无该符号（该模块只有 code-MILP `certify_css_code_distance:18`、exact-stim `exact_circuit_distance_from_pair:92`、DEM-ISD `isd_circuit_distance_from_pair:132` 三条通道）。实际落点是 `sec/schedule_rank.py`（`extended_matrices:151`、`d_ext_probe:164`，作为排名 stage 3 的顺带产物 `:228-231`）+ `evaluation/evidence.py`（`sec_ext_evidence:20-41`，以及与距离证据的合并 `:151-170`，`degraded_ext` 与 `verified_circuit_distance = min(exact-stim, DEM-ISD, d_circ_ext)` 均在此）。之所以没成为 `distance.py` 的独立通道：它是排名头部组的副产品，与选中调度同生共死，拆出去等于重算一遍。语义侧与 §5 完全一致（X/Z 分别记录、`degraded_ext` 对齐 `degraded_isd`、各通道取 min），只是地址要改读；篇首「替代关系」条（第 13 行）与 §12 第 6 步同此。

6. **§7.1 的调用方清单已随 03 稿重构作废**：`slow.py`、`postsearch.py`、`sinter_sweep.py` 三个文件都不复存在（slow loop 拆入 `evaluation/`）。`build_memory_experiment_pair` 当前的调用方是 `cli.py:758`、`qualification.py:147`、`export_stim.py:68`/`:73`、`evaluation/worker.py:67`/`:199`、`evaluation/pipeline.py:186`，另有 `verification/__init__.py:5` 的再导出。入口名与位置确实保留了，但签名并非原封不动：新增 `h_mask`/`deformation`/`sector_sizes`/`synth_cache_dir` 四个关键字参数，其中 `sector_sizes` 已是事实必填——缺失或长度不为 2 直接 `ValueError`（`verification/circuit.py:160-161`），因为 LRC 综合离不开双 sector 边界；`prefer_coloring` 仅为签名兼容保留、实现里忽略（见函数 docstring）。

7. **§3.4 四项协议参数的实际落点**：P 与探测预算是 `verification/circuit.py:86-87` 的模块常量（`LRC_SEAM_CANDIDATES = 64` / `LRC_SEAM_ISD_ITERS = 100`），不是 run config 字段，但确实进了 campaign 协议快照（`workflow/state.py:135-138`）——冻结与 resume 拒绝的效果达到了，可调性没有。`sec_policy_id` 进协议快照（`:132`）与 memory experiment metadata（`sec/schedule.py:96-98` → `verification/circuit.py:43`、`:73-74`），但**不进** slow_results 行：`RESULTS_CSV_COLUMNS` 里 sec 相关列只有 `deformation_mode`/`h_mask_weight`/`h_mask_sha256`/`d_circ_ext`/`degraded_ext`/`verified_circuit_distance`，无 `sec_policy_id`、无 P、无 `isd_iters`。`RankedSchedule.metadata`（`candidates`/`survivors`/`isd_iters`/`seed`/`chosen_candidate_index`，`sec/schedule_rank.py:241-247`）目前无任何消费方——`sec_ext_evidence` 只搬标量字段。§3.2 第 4 档的「d_ext 全量复排开关」未实现（规格本身默认关，此处只记实现状态）。

8. **§3.1 的「置换空间不足时取全部」未实现**：`rank_schedules` 恒定枚举 `max(1, candidates)` 个候选（`sec/schedule_rank.py:204`），`counter_permutation`（`sec/coloring.py:100-114`）对每个 (候选序号, 子图 slot) 独跑一次 splitmix64 驱动的 Fisher–Yates，既不去重也没有穷举分支；四子图的联合置换空间小于 P 时，候选里会出现重复调度，而不是改抽全部。确定性与「候选 0 = canonical 恒等序」两条按稿成立，受影响的只是小码上的有效候选数会低于 64。

9. **§6.2 ⑤ slope_offset 的落点**：实现为 `VerificationConfig` 的字段（`config.py:400`）而非 profile 注册声明的一项；「非 uniform 一律 None」的语义由构造侧保证——`cli.py:623` 只在 `uniform_legacy` 下给 0.958、其余给 None，`config.py:412-413` 再兜底清掉非 uniform 下残留的 ver1 默认值 0.958。语义等价，位置不同。

10. **本稿内部勘误**：篇首修订记录（第 5 行）的「§4.5 Manes–Claes 作用域收窄」没有对应小节——§4 只有 4.1–4.4；该收窄实际写在 **§3.3**（第 92 行）。同行其余三处指向（§2.1 / §5 / §1）均正确。

11. **Zenodo DOI 号在文档与代码间不一致**：第 206 行写 `10.5281/zenodo.18853600`，代码与回归则一致地写 `...18853601`（`sec/schedule.py:5`、`sec/stim_builder.py:4`、`tests/test_sec_golden.py:4`）。两者仅差末位，形态上像 concept DOI 与 version DOI 之别，但本次未联网核对 Zenodo 记录本身，故只记差异、不判孰是；CC-BY-4.0 的 license 结论与黄金样本可用性不受影响，需要对外引用时以代码中的号为准并另行复核一次。

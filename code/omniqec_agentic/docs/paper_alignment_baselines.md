# 对照方法与论文对齐说明：BO baseline、Random、消融、Fig.4 c 固定权重搜索

分支 `os/baselines`（基于 `opensource/paper-aligned` = release/final c6b4ffe6 + baselines/nonllm 847a6457），2026-11-10。

原则：论文是规格书。论文写了的，代码照做；论文没写的，保持现状；有歧义时取最字面的读法，并在下面写明。
引用位置：`L` = 论文纯文本 `paper14/plain.txt` 的行号，括号里是论文页边行号；正文 Methods "Baseline comparisons"
在 L1121–1150（页边 717–744）。

终局（前 5 名补测、Table VII）和 racing 归 `os/measure`，快环检查本身（合法 CSS、k ≥ 1、Tanner 图连通、
w_max ≤ 8）归 `os/search`。本分支只改基线、消融和固定权重的配置，以及让它们走共用路径所需的接线。

---

## 1 BO baseline（附录 D4）

| # | 论文要求（位置） | 代码现状 → 改动（文件:函数） | 歧义怎么定的 |
| --- | --- | --- | --- |
| 1 | 每轮随机生成 3,840 个 BB 提案（L4585，页边 1980） | 4 代 × 960（`config.BASELINE_PROPOSALS_PER_GENERATION/GENERATIONS_PER_TURN`），一致，未改 | — |
| 2 | ℓ、m 满足 4ℓm ≤ N（L4586–4587，1981–1982） | 当时：`sampler._bb_catalog` / `_draw_one` 先在可行的 R = 4ℓm 中均匀取，再在 ℓ ≤ m 的因子对中均匀取，未改。**11-10 已改为在全部可容许对上均匀、项数 (\|A\|, \|B\|) 在可容许拆分上均匀，见 §7** | 当时按论文只给约束、没给分布而保持现状；11-10 用户定按字面读法。ℓ ≤ m 只是去掉 C_ℓ×C_m ≅ C_m×C_ℓ 的重复标签 |
| 3 | A、B 项数合计 4–8（L4588，1983） | **原为 4–9**（经 `construction/bb.py` 的 `BB_MAX_TOTAL_WEIGHT = 9`）。**改**：`baselines/sampler.py` 新增 `BASELINE_BB_MAX_TOTAL_WEIGHT = 8`，`_bb_weights` 用它；`PRIOR_CONSTANTS["bb_total_weight"] = [4, 8]`；`SAMPLER_ID` `fixed_prior_v1 → fixed_prior_v2`（快照里的 catalog digest 随之改变）。LLM 用的 BB 构造器上限 9 **不动** | Random 的 BB 也用这个采样器，同样变成 4–8。D5 没写 Random 的 BB 项数，但写了 Random 受同样的最大校验权重约束；BB 码的行重恰好是 \|A\| + \|B\|，W = 9 的码权重必为 9 > 8，永远不合格，所以 4–8 是两个基线唯一一致的读法 |
| 4 | 项从 C_ℓ×C_m 里选（L4588） | 每个多项式内无放回均匀抽，一致，未改 | A、B 之间可以重叠（论文未限制） |
| 5 | 与 OmniQEC 相同的快环检查与筛选（L4589，1984） | 已走共用路径：基线程序进 `funsearch.run_funsearch` 的同一个 batch → `ArchiveMemory.evaluate_batch` → `construct_stage` → `structural_check_spec`；候选池（`selection.candidate_pool`）只读 `structural_ledger.jsonl`，里面只有结构合格的码，再加与 LLM 慢环队列相同的晋升门槛（d_search ≥ 5、w ≤ gate）。未改，加了测试 | `os/search` 改快环检查后基线自动继承，没有另一套检查 |
| 6 | 合格候选跨轮累积成池（L4589–4590） | 累积池，已选过的不再选，一致 | — |
| 7 | 池由随机生成的提案组成（L4585–4590）；手写种子程序只用于 LLM 运行的初始化（D1，L4229–4234，页边 1865–1870） | **原来每轮都跑 5 个手写种子程序**（BO 只有 BB 种子 [[30,4,2]] 过族过滤，d_search < 5，从不进池；Random 在 N = 98 / 180 / 260 时元循环种子 [[42,4,5]] w5 都能进池，可被均匀挑中）。**改**：`campaign.cycle_uses_seed_programs`，`campaign.run_agentic_cycle` 对 `search_method != "llm"` 不跑种子 | 论文没明说"基线不用种子"，但 D4 的池指"生成的码"，D5 说 Random 不用发现记忆、全部 115,200 次尝试都是随机抽样，而种子的作用（演示格式、初始化发现记忆）只对 LLM 有意义。取字面读法：基线不含种子 |
| 8 | 输入 H_X、H_Z 加代表性逻辑算符 L_X、L_Z；三视图（L4591–4592） | `bo_graphs.code_graph`：decode / xlogic / zlogic，一致 | — |
| 9 | 4 层关系图网络、128 维隐表示、按节点类型池化得 128 维嵌入（L4593–4595） | `BO_CONFIG.embedding`：d_model 128、num_layers 4、按类型均值池化后读出到 128 维，一致 | — |
| 10 | Matérn-3/2 + ARD + 线性均值；无预训练权重，联合优化（L4596–4598） | `bo._build_model`，一致 | — |
| 11 | 30 轮，Memory / CNOT 最多 180 / 210 个新候选（L4599–4600） | 30 × 6 / 30 × 7。**`--turns` 原默认 3**。**改**：`config.BASELINE_TURNS = 30`；`cli.build_parser` 的 `--turns` 默认 None，`cli.run_campaign_command` 对基线取 30、LLM 仍取 3；PBS 默认 `TURNS=30`（原本如此） | — |
| 12 | 目标 −log₁₀ p_LER；每基 Jeffreys Beta(f + ½, M − f + ½)，4,096 个后验样本，均值作训练目标、方差作观测噪声（L4601–4607） | `bo.jeffreys_objective`：p_LER = 1 − [(1 − q_X)(1 − q_Z)]^{1/(bkT)}，b = 2（CNOT，即 K = 2k）、T = R = 12，一致 | — |
| 13 | GP 另学残差噪声（L4607） | `FixedNoiseGaussianLikelihood(learn_additional_noise=True)`，一致 | — |
| 14 | AdamW 每轮最多 80 步，10 步无改进早停（L4655，2002） | **原为"10 步无改进且最后一步相对改进 < 1e-4"**（作者代码的规则）。**改**：`bo._train` 只看连续无改进步数；`BO_CONFIG.training.early_stop = {"patience": 10, "min_delta": 1e-6}`；偏差列表加第 8 条；`BO_ID` `bo_adapted_v1 → bo_adapted_v2` | "改进"定为损失下降超过 1e-6（原代码的判定）。两者只在 \|loss\| < 1e-2 时可能不同 |
| 15 | 从上一轮热启动（L4656） | 读上一轮的状态文件，一致 | — |
| 16 | 前 4 轮均匀随机挑，第 5 轮起 qLogNEI（L4660–4662） | `init_turns = 4`（第 0–3 轮），一致 | — |
| 17 | 256 个 Sobol 样本（L4663） | `acquisition.mc_samples = 256`，SobolQMCNormalSampler，一致 | — |
| 18 | Memory 每轮 6 个、CNOT 7 个（L4663–4664） | `slow_candidate_cap`：v3.7 结构配置 6、CNOT 配置 7，一致 | — |
| 19 | 顺序挑选，已选的作 pending（L4664–4665） | `bo._acquire`：`set_X_pending`，一致 | — |

论文没写、保持现状的：可用观测少于 2 个时退回均匀挑（并记录）、核长度尺度前 3 个 BO 轮冻结、ReduceLROnPlateau、
梯度裁剪、分组学习率、参考码的测量进入训练集（论文说"整个运行累积的电路级测量"）。

新增保护：`cli.run_campaign_command` 拒绝 `--search-method` 配 `--protocol-profile v3_8`。v3.8 的变体代在快环里只放行
父代的族，会把采样器的其他族提案当作 `family_not_allowed` 丢掉；基线只从结构配置取每轮名额（6 / 7）。

## 2 Random 与两个消融（Table VIII、附录 D5、Fig.10 / Fig.11）

Table VIII（L4612–4650）与代码的对应，逐项核对一致，**未改消融的定义**：

| 配置 | LLM 生成 | 码级反馈 | 电路级反馈 | 代码 |
| --- | --- | --- | --- | --- |
| Full | ✓ | ✓ | ✓ | `--ablation full`（默认） |
| w/o slow | ✓ | ✓ | × | `--ablation no_slow`：`feedback_mode = fast`，搜索期间不建慢环队列；planner、reflector、生成器只看快环信息 |
| w/o slow & fast | ✓ | × | × | `--ablation no_feedback`：`feedback_mode = none`，没有 planner、reflector、父代和归档上下文，每轮 4 次独立生成 |
| Random | × | × | × | `--search-method random --provider baseline_sampler`，四个码族；采样器不读任何反馈（`draw_generation` 的入参只有预算、码族、种子键和位置）；从合格池均匀挑 |

D5 的 Random 细节：

- 四族等分（L4709–4710）、每轮 3,840 次、每次运行 115,200 次（L4851）：每轮每族 960 次，一致。
- HGP 先抽两块种子矩阵的尺寸和校验权重，再抽元素；BB 抽群维数、两多项式的权重和支撑（L4709–4846）：一致。
- LP："先抽种子矩阵尺寸，由比特预算定可容许的群阶，再选该阶的阿贝尔 / 非阿贝尔群，最后抽元素的权重和群元素"
  （L4846–4848，页边 2045–2048）。代码是 R 优先：先在可行的总比特数 R 上均匀取，再在该 R 的（形状 A, 形状 B, 群）
  骨架中均匀取。当时**保持现状**：论文给了抽样对象和约束，没给每一步的分布；R 优先先验的联合分布可以分解成"先形状、再取
  阶可容许的群"，满足这段描述。另一种字面读法（先在形状上均匀，再在可容许阶的群里均匀）会显著改变已跑 Random 的
  分布，要用的话需另行决定。**11-10 用户已定按字面读法，四族都改为逐步均匀，见 §7。**
- 与 LLM 组相同的合法性约束（L4848–4849）：走共用快环（见 §1 第 5 条）。
- 每轮 Memory 6 个、CNOT 7 个，用同一慢环协议（L5632）：同 §1 第 18 条；racing 与当前最好的已测码和参考码比较
  （`graph._hof_thresholds`），与 LLM 组相同。
- 不用 LLM、发现记忆和性能反馈（L4852）：采样器同上；手写种子已去掉（§1 第 7 条）。

各组同一份默认协议（R = 12、w ≤ 8 门槛、30 次失败早停、同一模型）：

- 基线：`hpc/nscc_baseline_grid.pbs` 固定传 `--rounds-policy fixed_Tc --promotion-max-check-weight 8`。早停（交接自
  `os/measure` 26d00f27）：论文 D3 的早停是 run-campaign 不传 `--racing-max-errors` 时的默认策略
  `paper_d3_wilson10x_ref_maxerr30_v1`，传 30 / 100 会选旧策略，所以**删掉了原来的 `--racing-max-errors 30`**；
  `OMNIQEC_RACE_PILOT_WINDOW` 在论文早停下无效（第一次判定在 2,000 发试探之后），也一并删掉。测试锁定 PBS 里不再有这两项。
  在本分支单独看，不传该参数会落到旧默认 maxerr100；与 `os/measure` 合并后即为论文早停。
- 消融：**原来 PRESET 拒绝 ABLATION**，只能用历史协议启动。**改**：`hpc/nscc_claude_grid_v36.pbs` 的 PRESET 段接受
  `ABLATION=no_slow|no_feedback`（`fig2_final` 和 `fig3_final` 都可），其余键仍由 PRESET 设定；运行标签默认
  `f2final_no_slow` 等，不会写进主搜索的运行目录；`hpc/submit_claude_v36_grid.sh` 转发 ABLATION / EXACT_CHECK_WEIGHT /
  CAMPAIGN_ID，标签规则相同。论文配置：
  `PRESET=fig2_final|fig3_final ABLATION=no_slow|no_feedback VALIDATE_FORMAL=0 POINTS="98 180 260" bash submit_claude_v36_grid.sh`。
- PRESET=fig2_final 用 v3.8 结构配置，它每轮后 2 代是"变体代"，父代为已测最好的码加驻留参考码。w/o slow & fast 的定义是
  任何通道都没有父代，但在有参考码的预算点（N = 144、≥ 288）参考码仍会作为变体代父代传入。**改**：
  `workflow/graph.execute_plan_node` 在 `ablation == "no_feedback"` 时变体代父代为空（退化为探索代）。论文的消融点
  N = 98 / 180 / 260 没有参考码，不受影响；w/o slow 保持原样（参考码不是测量反馈，也从未被测）。
- 终局：消融从结构前沿取，基线从实测排名取（`workflow/graph._final_entries` 里各有一个分支），统一成前 5 名由 `os/measure` 做
  （26d00f27：`paper_top5_cumulative_v1`，基线和 Fig.4 c 都走它）。快照 `search_baseline` 里的描述字段 `final_selection`
  由 `measured_ranking_hof_v1` 改为 `common_final_stage_top5`（`cli.run_campaign_command`）：它只是描述，没有代码读取；实际
  规则由快照的 `final_stage` 键决定，合并后两者一致。

## 3 Fig.4 c 固定权重搜索（附录 D5，L4681–4686，页边 2017–2022）

| 论文要求 | 代码 → 改动 | 歧义 |
| --- | --- | --- |
| Claude，CNOT，N ∈ {98, 144, 180}，w_max 恰好等于 4、5、6、7，每个条件 1 次 30 轮运行 | `--exact-check-weight w`（`VerificationConfig.exact_check_weight`）。**改**：PRESET 接受 `EXACT_CHECK_WEIGHT`，限 `fig3_final`、3 ≤ w ≤ 8，慢环门槛同时取 w，运行标签默认 `f3final_w<w>`。论文配置：`PRESET=fig3_final EXACT_CHECK_WEIGHT=<w> VALIDATE_FORMAL=0 POINTS="98 144 180" bash submit_claude_v36_grid.sh`，w = 4..7 各一次 | 启动器允许到 8（主搜索的上限），论文的条件集是 4..7 |
| 与主搜索相同的程序生成和电路评估设置 | 由 PRESET 保证：T = 12、30 次失败、claude-opus-5 默认推理档 + 推理占满规则（论文 Table V：Claude CNOT 实际为 medium，这正是该规则的结果） | — |
| 候选码最大校验权重恰好为 w | 快环 `structural_check_spec` 判 `check_weight_mismatch`。**发现并修复**：并行快环（`OMNIQEC_FAST_WORKERS > 1`，NSCC 上默认开）的 worker 任务漏传 `exact_check_weight`，更轻的码能过快环、进归档和父代（慢环队列和终局的门槛仍会挡住）。**改** `memory/archive.py:ArchiveMemory.evaluate_batch` 的 `construct_jobs` | 交界处，见 §5 |
| 每个条件保留前 5 名（2026-11-10 论文修订后："Six candidates are evaluated in the final stage for each condition"，前 3 名报均值与极差） | 精确权重的运行走 `_paper_final_entries` 的通用（full）分支；11-10 起由精确权重约束自动选终局 `paper_fig4c_top6_confirm_v1`：前 6 个不同的码、每个 25 000 发、不补测、`final_ranking.json` 的 `reported_top` 给前 3。见 `paper_alignment_measurement.md` §8.2 | 6 个怎么选、测多少发论文没写，取主搜索同一规则与初始确认 |

Fig.4 c 与 S3 消融当时的部署脚本（env / submit / launch / rechain / audit / 顺序文件）记录的是在一个集群账号上的实际启动方式：
Fig.4 c 用 Fig.3 formal_v2 的 maxerr100、固定 medium、w = 4..8；S3 消融用 Memory T = max(12, d)、CNOT 权重上限 6、maxerr100。
它们与论文不一致，不作为论文配置，已从开源分支移除（只保留在本地）。

## 4 启动脚本与说明文档

- `hpc/nscc_baseline_grid.pbs`：文件头按论文重写（3,840 / 30 轮、BB 4–8 项、4ℓm ≤ N、四族等分、无种子、6 / 7 个、
  R = 12、w ≤ 8、论文早停（不传 `--racing-max-errors`）、通用终局前 5 名）；站点设置改为参数 `CONDA_ENV`（默认 omniqec，原写死 omniqec2）、`MODULES`
  （原写死 miniforge3/23.10 和 gcc）；BO 依赖检查去掉代码没用到的 torch_geometric。
- `hpc/submit_baseline_grid.sh`：`QUEUE`（默认 normal，原写死）、`CONDA_ENV`、`MODULES` 参数并转发；列出论文的运行。
  两个脚本里没有账号名，项目号和路径本来就是必填参数。
- BO 说明：`output/research_context/BO_baseline_implementation_20261107.md` 不在本分支（主工作树里未跟踪的文件），未改。
  它与论文不一致处：总项数 "4–9" 应为 4–8；"每轮照常跑 1 个手写 BB 种子程序" 已取消；早停条件见 §1 第 14 条；
  "终局按实测排名取前 6 个、25k 发" 将由 `os/measure` 的通用终局取代。与论文一致的版本即本文 §1。

## 5 与另外两个分支的交界

- `os/search`（快环检查）：基线池只读结构台账，`os/search` 的新检查自动生效。若 w_max ≤ 8 是用启动器参数而不是代码默认
  实现的，`nscc_baseline_grid.pbs` 也要加同样的参数（现在只有慢环门槛 `--promotion-max-check-weight 8`）。
  本分支动了 `memory/archive.py:evaluate_batch` 的 `construct_jobs`（加一个键）、`workflow/graph.execute_plan_node` 的 v3.8
  变体代父代（no_feedback 时为空）和启动器 PRESET 段，可能与对方冲突，冲突很小。
  手写种子（交接自 `os/search` 6a2b94c6 / 8dab1c3d）：`os/search` 让 `execute_plan_node` 只在第 0 轮传 `use_seeds`，基线的
  提案也走同一路径（`_baseline_sampler_batch` → `evaluate_batch`），w ≤ 8 快环检查对基线自动生效。本分支的
  `campaign.cycle_uses_seed_programs` 在 `run_agentic_cycle` 里把基线的 `use_seeds` 置为假，与调用方传入的值取"与"：合并后
  LLM 运行只在第 0 轮跑种子，BO 和 Random 任何一轮都不跑（§1 第 7 条）。这是基线专属的判断，不改共用的种子调用处。
  依据：论文只在 D1 说 OmniQEC 搜索第一轮用 5 个种子程序；D4 说 BO 每轮随机生成 3,840 个提案、池由这些生成的码组成；
  D5 说 Random 的 115,200 次尝试全部随机抽样、不用发现记忆。论文里找不到基线使用种子的依据。
- `os/measure`（终局）：`_final_entries` 有基线分支（`search_method != "llm"`，6 个全从实测排名取）和消融分支（结构前沿），
  统一成前 5 名时这两个分支都要覆盖（`os/measure` 已用 `_paper_final_entries` 覆盖）；`final_selection` 已改为
  `common_final_stage_top5`；`hpc/s3_ablation_top3.py` 读 `final_slow/T12/rankings.json`，`os/measure` 的终局写
  `final_slow/final_ranking.{json,csv}`，合并时要核对该工具的输入路径。
  `os/search` 也给并行快环补了 `exact_check_weight`，与本分支 `memory/archive.py` 的同一改动重复，合并时取一份即可。
- 协议 id 变了：`fixed_prior_v2`、`bo_adapted_v2`。已跑的基线运行（v1）不能用这份代码续跑或复算采样，这是有意的。
  11-10 采样器再改为 `literal_uniform_v1`，随后按用户决定改为 `literal_uniform_v2`（§7）；更早的运行都不能续跑。

## 6 测试

`tests/test_paper_alignment_baselines.py`：BO 提案 4–8 项且 4ℓm ≤ N、LLM 构造器上限不变；采样器不读反馈；基线提案经共用
结构检查、台账只含已检查的码；基线不跑手写种子；基线 `--turns` 默认 30、6 / 7 个名额；拒绝 v3.8；BO 设置逐项；Table VIII
对应；并行快环的精确权重；PRESET + ABLATION / EXACT_CHECK_WEIGHT；基线脚本无账号名；v3.8 下 no_feedback 的变体代无父代。另更新
`test_baseline_sampler.py`（≤ 8）、`test_baseline_selection.py`（sampler id）、`test_reasoning_fallback.py`（PRESET 不再拒绝
ABLATION；改为拒绝 Memory 上的精确权重）。BO 模型测试需要 torch / gpytorch / botorch，缺包时跳过。

## 7 2026-11-10：Random / BO 的字面逐步均匀抽样（`literal_uniform_v1` → `literal_uniform_v2`）

用户定："按字面'先在尺寸上均匀、再在群上均匀'来实现"。规则：论文点名的每一步都在该步的可容许集合上均匀抽，按论文的顺序。
可容许集合 = 构造前就能判定的预算和码族语法约束（在前面各步已定的条件下）；只有构造出的码才能判定的约束（CSS 合法、
k ≥ 1、Tanner 图连通、w_max ≤ 8）仍由共用快环拒绝；语法上用不了的随机参数记为失败，不重抽。

论文原文（附录 D5，Ablation 部分）：

> For HGP codes, we sample the dimensions and check weights of the two classical seed matrices, followed by their
> binary entries. For generalized BB codes, we sample the cyclic group dimensions, the weights of the two group-algebra
> polynomials and their monomial supports. For Abelian and non-Abelian LP codes, we first sample the seed-matrix
> dimensions and determine the admissible group orders from the physical-qubit budget. We then select an Abelian or
> non-Abelian group of an admissible order, respectively, and sample the weights and group elements of the seed-matrix
> entries.

附录 D4（BO）：

> For each proposal, we sample the two cyclic-group dimensions ℓ and m subject to 4ℓm ≤ N. We then randomly determine
> the numbers of terms in the two BB polynomials A and B, with a combined total between four and eight, and select the
> corresponding group elements from C_ℓ×C_m.

R = 数据比特 + 每个校验行一个辅助比特（构造器和结构检查数的预算），以下各集合都保证 R ≤ N。代码都在 `baselines/sampler.py`。

| 步骤 | 旧规则（`fixed_prior_v2`） | 新规则（`literal_uniform_v2`；与 v1 只差 LP 格权重） | 函数 |
| --- | --- | --- | --- |
| 族划分 | 每轮四族等分，打乱 | 不变（每次运行每族 28,800 次 = 115,200 / 4） | `turn_family_plan` |
| BB (ℓ, m) | 先在不同的 R = 4ℓm 上均匀，再在该 R 的因子对中均匀 | 在全部可容许对 {ℓ ≤ m, ℓm ≥ 2, 4ℓm ≤ N} 上均匀 | `_draw_skeleton`、`_bb_pairs` |
| BB 项数 | W = \|A\| + \|B\| 在 4..8 上均匀，再在拆分中均匀 | (\|A\|, \|B\|) 在可容许对 {1 ≤ \|A\|, \|B\| ≤ ℓm, 4 ≤ \|A\| + \|B\| ≤ 8} 上均匀（W = 4..8 的比例为 3:4:5:6:7） | `_bb_weight_pairs`、`_draw_bb` |
| BB 支撑 | 每个多项式在 Z_ℓ×Z_m 中无放回均匀 | 不变 | `_draw_bb` |
| HGP 尺寸 | 先在不同的 R = (r1+n1)(r2+n2) 上均匀，再在该 R 的形状对中均匀 | 在全部可容许有序形状对 {(r1+n1)(r2+n2) ≤ N，每个形状都有合法种子} 上均匀 | `_draw_skeleton`、`_hgp_pairs` |
| HGP 校验权重 | 边数 E 在可行范围上均匀，再抽和为 E 的行度序列和列度序列（各自均匀），Gale–Ryser | 行度序列（每行的校验权重）在可容许序列上均匀：每项 ∈ [2, min(4, n)]，和在列能承受的 [2n, min(4, r)·n] 内（精确计数，不拒绝） | `_check_weights` |
| HGP 元素 | 随机桩匹配，重边算失败 | 每行支撑在该大小的子集中均匀（= 给定行重的二元矩阵上均匀）；列度不在 [2, 4] 或有重复行 → 失败。不再单独抽列度序列（论文没有这一步） | `_seed_entries`、`_draw_hgp` |
| LP 形状 | 先在不同的 R = sA·sB·\|G\| 上均匀，再在该 R 的（形状 A, 形状 B, 群）中均匀 | (形状 A, 形状 B) 在 {1×1, 1×2, 2×1, 2×2}² 中至少有一个可容许阶的有序对上均匀（N = 98 / 180 / 260 时 16 对全可容许） | `_draw_skeleton`、`_lp_stages` |
| LP 阶 | （同上） | 可容许阶 = {2 ≤ \|G\| ≤ N / (sA·sB)，且目录里有该族该阶的群}，均匀取一个 | `_draw_skeleton`、`_lp_stages` |
| LP 群 | （同上） | 在目录中该族该阶的群里均匀；每个同构类一个（阿贝尔按不变因子 `abelian_catalog`，非阿贝尔 `nonabelian_catalog`） | `_draw_skeleton` |
| LP 元素 | 格权重 q(w) ∝ 2^-(w-1)，w ∈ 1..min(8, \|G\|)；元素无放回均匀 | v2：A、B 的格权重整体在可容许赋值上均匀——每格 ∈ 1..min(8, \|G\|)，且提升码的 w_max = max(行(A) + 列(B), 行(B) + 列(A)) ≤ 8（见下面的 11-10 用户决定）；然后每格元素无放回均匀；A、B 的支撑须生成 G（语法，失败不重抽）。v1 曾是每格独立在 1..min(8, \|G\|) 上均匀，w_max ≤ 8 交给快环 | `_draw_cell_weights`、`_cell_weight_table`、`_lp_max_check_weight`、`_draw_cells`、`_draw_lp` |

几点说明：

- **ℓ ≤ m**：BB 校验器把 ℓ > m 连同环面坐标对调成 ℓ ≤ m（`construction/bb.validate_bb_spec`），(m, ℓ) 与 (ℓ, m) 是
  同一个码，所以每个无序对算一个标签、在无序对上均匀。若改为在有序对上均匀，ℓ = m 的对只有其他对一半的权重。
- **可容许集合没有变**：每族的骨架集合与旧版逐个相同（N = 60、72、98、144、180、260、360 全部比对一致），变的只是集合上的
  分布和骨架内参数的分布。`resource_catalog` 保留为按 R 分组的视图；`stage_catalog` 给出各步实际抽取的集合，
  `catalog_digest` 改为对它取哈希。
- **目录里没有该族该阶的群**：阿贝尔族每个阶都有循环群，不会发生。非阿贝尔族若在 2..⌊N/(sA·sB)⌋ 的全部整数上均匀
  取阶，取到"没有非阿贝尔群"的阶的概率在 N = 98 / 180 / 260 时为 65.7% / 56.5% / 50.2%（形状对均匀，精确计算），
  一多半非阿贝尔名额会白费。取最小的忠实规则：可容许阶只含目录里有该族群的阶。到 |G| = 65（N = 260 时 1×1 形状的上限）
  为止，去掉的阶恰好是"该阶的群全是阿贝尔群"的阶（2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 25, 29, 31, 33, 35, 37,
  41, 43, 45, 47, 49, 51, 53, 59, 61, 65），目录不缺任何阶；到 |G| = 90 唯一的缺口是 75（(C5×C5)⋊C3 不在目录里），只在
  N ≥ 300 时出现，论文 Random 的三个点不受影响。平凡群（阶 1）不在目录里，一直不取。注意：这里是"先阶均匀、再该阶的群
  均匀"，不是"在所有可容许阶的群上均匀"；后者会偏向群多的阶（16、24、32、48、64）。
- **参数组整体均匀**：(|A|, |B|)、HGP 的行度序列和 LP 的格权重都是在整组的可容许集合上均匀，不是每个分量独立均匀
  （论文没说）。
- **BO**：BO 的提案就是同一采样器的 BB-only 抽样（`funsearch._baseline_sampler_batch` → `draw_generation(families=("bb",))`），
  所以 BO 的 (ℓ, m) 和 (|A|, |B|) 也按新规则，与 D4 的描述一致。`BO_ID` 不变（模型没变）。
- **SAMPLER_ID**：`fixed_prior_v2 → literal_uniform_v1 → literal_uniform_v2`（没用 `uniform_paper_v1`：那是噪声模型的
  id；v2 只改了 LP 格权重，见下）。它进入 seed key、
  每次抽样的 RNG 种子、`catalog_digest` 和快照的 `required_model` / `search_baseline`；旧运行续跑会被
  `workflow/checkpoint.validate_resume_protocol` 拒绝。

### 11-10 用户决定：LP 格权重并入 w_max ≤ 8（`literal_uniform_v2`）

用户定（歧义 1）：把 w_max ≤ 8 并入 LP 格权重的可容许集合，再在该集合上均匀抽，和 BB 的"合计 4–8 项"同一做法。理由：LP 的
w_max 完全由格权重决定（X 校验 (i, t) = A 第 i 行的格权重和 + B 第 t 列的格权重和，Z 校验 (j, s) = B 第 s 行 + A 第 j 列；
一个格内元素互不相同，提升后权重不变，见 `construction/lp.lp_code`），构造前就能判定。v1 测量中构造出的 502,271 个 LP
码，按格权重推算的 w_max 与构造值全部一致。

实现（`baselines/sampler.py`）：`_cell_weight_matrices` 按 (最大行和, 最大列和) 把一种形状的全部权重矩阵（每格
1..min(8, |G|)）分组；`_cell_weight_table` 列出满足 行(A) + 列(B) ≤ 8 且 行(B) + 列(A) ≤ 8 的 (A 组, B 组) 块及其大小；
`_draw_cell_weights` 先按块大小加权选块，再在块内均匀选 A、B，所以在可容许的联合赋值上精确均匀、不拒绝。之后元素照旧
（`_draw_cells`），A、B 支撑须生成 G 的规则不变。可容许赋值个数举例（|G| ≥ 8）：1×1 / 1×1 有 28 个，1×2 / 2×1 有 70 个，
2×2 / 2×2 有 1,526 个（`lp_cell_weight_count`）。`BASELINE_MAX_CHECK_WEIGHT = 8` 即快环的 `SEARCH_MAX_CHECK_WEIGHT`；
快环照样检查每个码。分布变了，所以 `SAMPLER_ID` 改为 `literal_uniform_v2`，v1 的运行不能续跑。

### 效果（构造层，不跑线路和解码器）

每个 (采样器, N) 抽 115,200 次（30 轮 × 3,840，每族 28,800），抽出的每个 spec 过快环结构检查
（`structural_check_spec`，总比特预算 N，w_max ≤ 8）；d 是快环协议的 d_search（种子 0–3、每种子 120 次迭代），在每格
不同的合格码里均匀抽 400 个（HGP 不足 400 时全算）；"估计 d ≥ 5 不同码" = 不同合格码数 × 样本中 d ≥ 5 的比例
（慢环晋升门槛 d_search ≥ 5）。

| 族 | N | 通过 % 旧→新 | 不同码 旧→新 | n 中位 (p25–p75) 旧 → 新 | w 中位 | d 中位 | d ≥ 5 % | 估计 d ≥ 5 不同码 |
|---|---|---|---|---|---|---|---|---|
| bb | 98 | 28.0 → 26.5 | 5701 → 6281 | 22 (8–34) → 28 (14–38) | 6 → 6 | 4 → 5 | 40 → 51 | 2280 → 3188 |
| bb | 180 | 25.8 → 25.3 | 6218 → 6683 | 42 (18–66) → 50 (26–72) | 6 → 7 | 5 → 6 | 65 → 68 | 4042 → 4578 |
| bb | 260 | 24.8 → 25.1 | 6363 → 6879 | 62 (28–96) → 72 (40–102) | 6 → 7 | 6 → 7 | 74 → 76 | 4677 → 5245 |
| hgp | 98 | 0.7 → 1.0 | 198 → 286 | 35 (27–42) → 39 (34–43) | 6 → 8 | 2 → 2 | 0 → 0 | 0 → 0 |
| hgp | 180 | 0.8 → 0.4 | 219 → 104 | 61 (39–77) → 53 (41–66) | 6 → 8 | 2 → 2 | 0 → 1 | 0 → 1 |
| hgp | 260 | 0.8 → 0.2 | 235 → 44 | 81 (51–110) → 63 (47–75) | 6 → 8 | 2 → 2 | 1 → 2 | 2 → 1 |
| lp_abelian | 98 | 22.7 → 46.8 | 5971 → 12455 | 25 (15–36) → 30 (18–40) | 6 → 7 | 3 → 3 | 9 → 5 | 537 → 592 |
| lp_abelian | 180 | 21.6 → 46.1 | 6056 → 12887 | 45 (25–70) → 50 (28–72) | 6 → 8 | 3 → 3 | 16 → 12 | 939 → 1514 |
| lp_abelian | 260 | 20.7 → 45.7 | 5881 → 12958 | 65 (34–100) → 69 (36–100) | 6 → 8 | 3 → 3 | 20 → 16 | 1147 → 2073 |
| lp_nonabelian | 98 | 16.0 → 48.1 | 4577 → 13839 | 30 (24–42) → 36 (32–48) | 6 → 8 | 2 → 2 | 2 → 2 | 103 → 277 |
| lp_nonabelian | 180 | 15.5 → 47.2 | 4464 → 13581 | 52 (30–72) → 60 (40–78) | 6 → 8 | 2 → 2 | 6 → 5 | 290 → 713 |
| lp_nonabelian | 260 | 15.8 → 47.5 | 4550 → 13676 | 76 (44–108) → 84 (54–108) | 6 → 8 | 2 → 3 | 8 → 6 | 375 → 855 |

BB、HGP 两行是 `literal_uniform_v1` 的测量（v2 只改了 LP，BB / HGP 的分布不变）；LP 行的"新"是 `literal_uniform_v2`
（另跑一遍 115,200 次，同样的检查和 d 抽样）。v1 的 LP 结果留作记录：通过率阿贝尔 13.9 / 8.0 / 6.2%、非阿贝尔 3.3 / 2.7 /
2.3%，估计 d ≥ 5 不同码阿贝尔 97 / 157 / 197、非阿贝尔 26 / 104 / 92（N = 98 / 180 / 260）。

拆开看（同一批抽样）：

- 只换尺寸 / 群这几步、族内参数仍用旧规则时，LP 通过率略升（阿贝尔 28.6 / 26.1 / 24.9%，非阿贝尔 22.9 / 22.1 /
  22.4%），BB 基本不变（25.9 / 24.1 / 23.4%）。v1 的 LP 大幅下降全来自格权重每格独立均匀：w_max > 8 被快环拒绝的占 LP
  尝试的 42–62%（旧规则 5.7–7.6%）。v2 把 w_max ≤ 8 并入可容许集合后，LP 不再有 w > 8 的拒绝，通过率约 46–48%，主要的
  剩余拒绝是 k = 0（阿贝尔约 51%、非阿贝尔约 47% 的尝试），抽样失败（支撑只生成真子群）0.6–1.1%。形状对均匀后
  1×1×1×1 只占 LP 尝试的 1/16（旧规则 37–41%）。
- BB 的 n 和 w 变大：有序对计数偏向因子多的 ℓm，项数拆分均匀偏向 W 大的一侧。d ≥ 5 的不同码略多。
- HGP 两版都几乎不产生 d ≥ 5 的码。新规则的失败主要是列度不在 [2, 4]（N = 260 时 25,497 / 28,800），旧规则主要是桩匹配
  出重边（27,074 / 28,800）。
- LP 抽样阶段失败（支撑只生成真子群）：旧规则 3–11%，v1 0.2%，v2 0.6–1.1%（格权重越重越容易生成整个群）。

### 歧义

- **LP 格权重的可容许集合**：11-10 用户已定，并入 w_max ≤ 8（`literal_uniform_v2`，见上）。更正：v1 版本这里曾把"以 w ≤ 8
  为条件"的通过率（阿贝尔 49.5 / 46.4 / 45.4%、非阿贝尔 40.8 / 38.2 / 35.0%）当作该做法的预测。这只在每个骨架内成立：
  按 w ≤ 8 取条件也会改变形状 / 阶 / 群的权重，而 v2 的这几步仍然是均匀的。v2 的实测值见上表。
- **HGP 元素**（保持现状，用户定）：按字面"给定行重、二元元素均匀"实现，列度约束由语法拒绝。另一种读法是元素在满足列度
  约束的矩阵上均匀（需要重抽或计数），论文没说。两版 HGP 都对 Random 池几乎无贡献。
- **参数组整体均匀**（保持现状，用户定）：见上面"几点说明"。

### 测试

新增 `tests/test_baseline_sampler_literal.py`：SAMPLER_ID 变更且旧快照续跑被拒；digest 和抽样随 SAMPLER_ID 变；每次运行
每族 28,800 次；BB (ℓ, m) 集合逐个枚举 + 卡方检验均匀（也覆盖 BO 的 BB-only 抽样）；(|A|, |B|) 拆分枚举 + 卡方；BB 支撑
均匀；HGP 形状对枚举 + 卡方；HGP 行度序列在小种子上穷举可容许序列 + 卡方；行支撑均匀；LP 各族的形状对、可容许阶、
每阶的群集合逐个枚举（非阿贝尔去掉的阶 = 全阿贝尔阶，到 |G| = 90 只缺 75）；LP 形状、阶（给定形状）、群（给定阶）分别
卡方检验均匀；LP 格权重的可容许联合赋值与穷举一致（7 组形状 / 阶）、在其上卡方检验均匀，每次抽样推算的 w_max ≤ 8 且与结构检查的值相同；群元素均匀；LP spec 的 R 与构造一致。卡方检验阈值 p > 1e-4（抽样是确定的，测试要么总过要么总不过），
已确认旧规则在同样的检验下 p 值为 0 或 1e-156 量级。更新 `test_baseline_sampler.py`（失败原因；格权重不再递减，v2 每格最多 7）、
`test_paper_alignment_baselines.py` 和 `test_baseline_selection.py`（sampler id）。

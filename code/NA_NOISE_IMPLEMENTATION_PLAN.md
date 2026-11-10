**OmniQEC 中性原子噪声实现方案**

日期：2026-09-07。状态：供实施的设计稿，未实现代码，未运行验证。目标位置：仓库根目录的 `NA_NOISE_IMPLEMENTATION_PLAN.md`。

本方案以本次讨论及先前成功读取的项目代码、设计文档为依据。本轮重新读取项目时，操作系统返回 `Operation not permitted`，因此下文的现有接口定位是前次读取的快照，实施前需要核对当前版本。所有“建议新增”的文件、字段、接口和命令均是拟议设计，不代表仓库已有实现。

**1. 要实现什么，以及怎样判定完成。** 首个目标是一个明确标注假设的、受中性原子实验启发的电路级模型。它包含独立配置的操作错误、持续 atom loss、实际可见的损失标记，以及能使用这些标记的译码与计分。第二个目标是检验这些机制是否改变候选码的相对表现。不能把“必须出现超导和 NA 各自的赢家”写成实现验收条件。

建议拆成两个 profile，并共享一份操作噪声配置：

| 建议 profile 名 | 内容 | 用途 |
| --- | --- | --- |
| `na_pauli_v1` | CZ-native，操作类别分别配置 Pauli 噪声，无 loss | 实现起点、现有后端兼容性检查、无损失消融 |
| `na_rb_loss_v1` | 相同操作噪声，加持续缺位及延迟检测 | 首个 NA loss 主模型候选 |

`na_rb_loss_v1` 首版范围冻结为：Rb-inspired、状态无关的独立 loss、抽象 all-to-all、固定 SEC 调度、ancilla 在读出时报告 loss、data 在终局读出时报告 loss。data 在实验期间不补充；ancilla 在检测到丢失后、下一次使用前显式补充。先使用理想的损失识别与 ancilla 补充作为声明过的简化。

这是有限轮数的逻辑记忆实验。仅在终局发现 data loss、且不修复 data 的设置不能支持长期稳定量子存储的结论。首版结果应称为“在指定噪声、检测协议与调度抽象下的码表现”，不能泛化成所有中性原子设备的表现。

首版不实现 movement/layout、真实 LDU、态依赖 loss、Rydberg leakage 动力学或相关双原子 loss。这些都应有独立扩展位置，但不先堆入主模型。不得把 Yb 的 metastable erasure-conversion 参数直接混入 Rb profile。

**2. 与项目现状的关系。** 前次代码和文档快照显示：正式搜索协议为 `telescoping_structural_v3_6`，使用 `uniform_paper_v1`、`p=0.005`、`twobasis_nonint_v1`、T=12；当前有 uniform、SI1000 和 Helios，ATOM 标签尚未对应独立的完整噪声评估路径。Helios 是离子阱模型，不能作为 NA 参数来源。

因此，NA 应使用新的实验协议身份。现有 uniform 正式结果、续跑检查与缓存命名不能被新参数静默改变。两基每轮均测、非交错的 SEC 可以作为第一版比较的公共结构；native H/CZ 发射与 idle 时间线需要核对。当前文档中 SI1000/Helios 在这一时间线上的硬件优化尚未完成，不能把其代码简单复制后视为硬件模型已校准。

**3. 参数契约：把事件概率、实验保真度和模型假设分开。** 建议新增冻结的 NA 配置对象，所有物理字段都参与 canonical serialization 与配置哈希。第一版运行要求显式提供概率，不能把遗留 `p` 静默解释成所有参数。

| 建议字段 | 精确定义 | 首版处理 |
| --- | --- | --- |
| `p_cz_pauli` | 每次计划 CZ 脉冲对应的 D2 非恒等 Pauli 总概率 | 独立参数；D2 的 15 项各为 p/15 |
| `p_1q` | 每个实际执行的单比特物理门后的 D1 总概率 | H 也计入；虚拟 frame 操作是否免费要明确 |
| `p_prep` | 在场原子初始化到目标基态后的错误概率 | Z 初始化后 X flip；X 初始化的实现见下文 |
| `p_meas` | 原子在场时，报告的二值读出结果翻转概率 | 与是否存在原子的识别错误分开 |
| `loss_per_atom_per_cz` | 在场原子每次计划 CZ 脉冲的 loss 概率，记为 ℓ | 首版独立、状态无关 |
| `idle_model` | `none` 或 `time_dephasing` | 必须显式选；none 是忽略等待误差的简化 |
| `idle_Tphi_us` | 纯退相位的相干衰减时间 | 仅 time_dephasing 时必填 |
| `duration_*_us` | 1Q、CZ、初始化、读出、补充等操作时长 | 时间模型开启时必填，不使用隐式单位 |
| `loss_detection` | ancilla 读出、data 终局读出 | 固定协议枚举，纳入身份 |
| `reload_policy` | 检测到缺失的 ancilla 在下次使用前补充 | 第一版理想成功，补充事件仍记录 |
| `missing_partner_channel` | 计划 CZ 的一个端点缺失时，幸存端的脉冲噪声 | 首版 `d2_marginal` |
| `loss_position` | CZ 脉冲的离散 loss 插入位置 | 首版 `before_ideal_cz` |
| `calibration_status` | `synthetic`、`paper_inspired` 或 `device_calibrated` | 禁止把示例配置标成实测校准 |
| `provenance` | 来源、版本、操作单位、条件化口径、转换公式 | 每类参数分别记录 |

数学约定：

\[
\mathcal D_2(p)(\rho)=(1-p)\rho+\frac{p}{15}\sum_{P\in\{I,X,Y,Z\}^{\otimes2}\setminus\{II\}}P\rho P.
\]

\[
\mathcal D_1(p)(\rho)=(1-p)\rho+\frac p3(X\rho X+Y\rho Y+Z\rho Z).
\]

loss 与 Pauli 使用顺序复合的随机过程，允许同一次脉冲中发生 loss 并对幸存原子施加 Pauli；二者不是互斥类别。因此，不要求 `p_cz_pauli + ℓ <= 1`，而要求每个通道本身合法。若未来改为互斥 mixture，应另立模型版本，不能悄悄改变语义。

在两原子均在场、loss 独立的假设下，一次脉冲至少损失一个原子的概率为 `1-(1-ℓ)^2`。这个 pair-level 概率、单原子 ℓ、平均门不保真度和 D2 参数不能混用。

**4. 初始数值与校准路线。** 可核实的实验锚点是 Evered 等人 2026 年预印本：每原子每 CZ 的 loss 为 `0.054(2)% = (5.4 ± 0.2) × 10^-4`；raw CZ fidelity 为 99.854(4)%，loss-postselected fidelity 为 99.941(3)%。论文报告 loss 约占门错误的 60%，并观察到双原子 loss 相关性。独立 ℓ 模型只近似其边缘 loss 率。[来源：实验正文的 Benchmarking low-error CZ gates](https://arxiv.org/html/2604.25987v1)

这一来源并不提供本方案已经核实的完整 1Q、读出、初始化、等待、补原子参数包。不能声称下方所有示例数字都来自该实验，也不能把 raw infidelity 全部转成 Pauli 再额外叠加 loss。

如果把 loss-postselected 的条件通道近似成各向同性 D2，并进一步把其拟合 fidelity 当作平均门保真度，那么 d=4 时 `r = 4p/5`，对应 `p ≈ (5/4) × 0.00059 = 7.375 × 10^-4`。这只是带假设的有效参数换算，不是实验直接测得的 D2 强度；条件 benchmark 也不能确定缺失伙伴时的噪声。第一版不把这一推算硬编码成通用默认值。

实现时可先使用下面的纯工程连通性配置。除 ℓ 借用量级锚点外，其他数值是本方案选定的测试值；此配置不得用于声称实测平台优劣：

```yaml
# 建议新增 schema，当前仓库尚不保证支持。
profile: na_rb_loss_v1
protocol: na_memory_pilot_v1
calibration_status: synthetic
rounds: 2  # 微型电路连通性测试；研究阶段再显式设为 12
sec_template: twobasis_nonint_v1
connectivity: abstract_all_to_all

pauli:
  cz_model: depolarize2
  p_cz_pauli: 0.001
  p_1q: 0.0001
  p_prep: 0.001
  p_meas: 0.002

idle:
  model: none

loss:
  per_atom_per_cz: 0.00054
  independent_atoms: true
  state_independent: true
  position: before_ideal_cz
  missing_partner_channel: d2_marginal

readout:
  ancilla: each_scheduled_readout
  data: terminal_only
  presence_detection: ideal

reload:
  target: detected_missing_ancilla
  boundary: before_next_use
  success_probability: 1.0
  duration_us: 0.0
  initialization_error: use_p_prep

sampler: explicit_loss_trajectory_v1
decoder: loss_ml_reference_v1
metric: logical_memory_classification_v1
```

上例采用 T=2，`decoder` 是小规模正确性基准，只用于可枚举的微型测试电路，不能直接拿去运行大规模搜索。`na_pauli_v1` 应使用完全相同的操作参数而关闭 loss；保持操作发射政策相同。研究阶段的 T=12 需在可扩展 loss 后端通过验证后另行配置。

探索性网格建议先单轴扫描 `ℓ ∈ {0, 1e-4, 5.4e-4, 1e-3, 2e-3}`，固定其他参数。这是敏感性范围，不是置信区间，更不是同一设备测得的五个工作点。对 CZ Pauli、测量和等待参数使用独立扫描；选择正式 profile 时，应在查看最终赢家前冻结来源与范围。

不要将 SI1000 的 `p=0.005` 与这个现代 ℓ 的组合标成同设备校准。也不要把论文中的“loss 占不保真度 60%”直接设为某个每原子随机 mixture 的 `loss_fraction=0.6`。只有分母、事件定义和标定口径一致时，fraction 才能这样转换。

**5. Native 门、测量和等待的实施规则。** 抽象 CSS SEC 首先生成确定的操作序列，再执行 hardware lowering；CZ-native 版本通过目标位 H、CZ、目标位 H 实现 CX。对真正执行的物理 H 加 1Q noise。噪声在 lowering 与合法的理想门化简之后放置，不能穿过噪声通道直接删除两个 H。

Z 初始化使用实际 reset 加初始化错误。若 X 初始化采用 `RZ + H`，则初始化误差和 H 误差分别计入；不要同时再插一份原生 RX 初始化噪声。X 读出若由 H+MZ 实现，则 H noise 与经典 readout flip 各计一次。能否将某个转换吸收到设备原生操作中，应由 profile 明确声明并固定。

读出生成 `0 / 1 / LOST`。第一版 `p_meas` 仅翻转 0 和 1；LOST 不经相同的二元 flip。以后加入存在性漏检、误报时，需独立字段和观测模型。

若开启时间等待模型：每个物理 layer 有实际时长，每个在场原子按该 layer 中未被占用的时间累计等待。明确读出等待与补原子等待是否使其他原子暂停，不用 Stim 的 TICK 数量冒充实际时间。纯退相位可采用

\[
p_Z(\Delta t)=\frac{1-e^{-\Delta t/T_\phi}}2,
\]

对应非对角元按 `exp(-Δt/Tphi)` 衰减；这里不是包含 T1 的完整 T2 模型。无参数来源时选择并标记 `idle_model=none`，不能称中性原子没有 idle noise。不要把 SI1000 的 measurement/reset extra-idle 参数原样移植到 NA。

首版调度不依据模拟器掌握的隐藏 loss 时刻改变。检测前仍按计划发出门脉冲；只有已经观测到的信息才能触发控制策略。现阶段的 all-to-all 无移动成本假设必须写进结果。

**6. 持续 loss 的物理与状态语义。** 对在场原子 q 和其余系统 R，状态无关 loss 的丢失分支是

\[
\rho_{qR}\longmapsto |\varnothing\rangle\langle\varnothing|_q\otimes\operatorname{Tr}_q\rho_{qR}.
\]

因此不只是“删除未来的 CZ”。必须丢弃该原子携带的量子信息，并保留其余系统正确的约化态。稳定子轨迹模拟可以用不可见的测量及 reset 实现 partial trace 的采样分解，但相关随机结果只能属于模拟器内部，不能成为 decoder 的输入。reset 出来的 dummy qubit 不代表原子已被补回。

建议分别记录 `present[q]`、`detected_missing[q]`、`atom_generation[q]`。发现 loss 只更新观测知识；实际 reload 才把 `present` 改回 true，并增加 generation。普通 reset 只能初始化在场原子，不能复活缺失 ancilla。

Perrin 等人的理论模型采用独立的逐 CZ loss，并在发生 loss 后取消当前和后续相关理想门，同时保留对幸存伙伴的脉冲噪声。首版借鉴这一离散近似，不声称精确描述 CZ 脉冲中任意时刻的物理损失。[来源：Error models](https://arxiv.org/html/2412.07841v3)

每个计划 CZ(a,b) 的处理顺序固定如下：

1. 对脉冲前仍在场的每个端点独立抽样 Bernoulli(ℓ)。另一端已经缺失时，在场端仍承受该计划脉冲的 loss 风险。
2. 对新丢失端执行 trace/discard，更新 occupancy。首版在 ideal CZ 前处理 loss；不要先执行完整 CZ 再说这是相同模型。
3. 两端都在场才执行 ideal CZ。只有一端在场时不执行该理想纠缠门；两端都缺失则不作用于量子态。
4. 两端在场：施加 D2(p_cz_pauli)。只有一端在场：施加预先声明的幸存端通道。两端缺失：无量子 Pauli 作用。
5. 不因为模拟器在第 1 步知道 loss，就立即产生实验 herald 或提前改变后续调度。

`d2_marginal` 定义为：将 D2 的两位 Pauli 分布对缺失端求边缘分布。幸存端每种非恒等 Pauli 有 4p/15 的概率，因此对应 D1(4p/5)。这是所选 D2 近似的数学延伸，不是缺失伙伴时的实验标定。未来替换为明确的 `pX,pY,pZ` 或 loss-correlated Z 通道时，另设参数与模型版本。

计划的单比特门对缺失位置不执行；在场位置照常执行并加噪。读出时，缺失位置输出 LOST；在场位置量子测量后施加经典读出 flip。读出 ancilla 的补充发生在之后的明确边界；若同一物理位置在一轮内多次读出与复用，按其实际生命周期处理，不能假定永远“一轮一次”。

首版的零耗时、必成功 ancilla reload 是理想化协议部件；仍记录次数和初始化误差。以后开启失败率或实际补充耗时，必须同时修改调度、数据等待和观测模型。data 不经正常 SEC reset；终局缺失 data 不偷偷补原子。

一个原子在某个生命周期内有 m 次在场时可能承受的计划脉冲，独立常数 ℓ 模型给出生存概率 `(1-ℓ)^m`。该量用于验证 loss 采样，不是逻辑错误率。持续缺失可能影响多个后续检查，但已丢失的原子不能重复计成新的 loss 事件。

**7. 模拟器真值与 decoder 可见信息必须分开。** 建议新增三类数据对象，名字可在实施时调整：

```text
LossTruthTrace                    # 仅模拟与验证内部可用
  shot_id, atom_generation, exact_loss_event_id
  hidden_discard_outcomes, hidden_pauli_draws
  occupancy_history, private_logical_reference

ObservedMemoryRecord              # decoder 可见
  shot_id
  measurement_slots: 0 | 1 | LOST
  measurement_metadata: qubit, basis, scheduled_time
  public_reload_events
  circuit_and_protocol_identity

MemoryDecodeResult
  output_kind: logical_labels | observable_frame
  predicted_logical_bits_or_frame
  status: decoded | declared_failure | unsupported
  model_version, approximation_metadata
```

禁止把 exact loss event、隐藏测量结果、真实 Pauli 错误或原始逻辑标签传给 decoder。公开的 `shot_id` 只用于结果关联，decoder 不得由它恢复随机种子或抽样轨迹。worker 随机种子和 truth 文件不属于译码接口。

首版 ancilla 的 loss 在其读出时可见；data 的 loss 在终局可见。终局发现 data 丢失，意味着它可能在整个未检查区间的任意合适位置丢失，不能把 loss 时刻自动设为最后一轮。更不能把即时 herald 的独立标记简单挪到周期末后，仍给 decoder 保留每个真实故障位置的身份。[延迟检测与生命周期的研究依据](https://arxiv.org/html/2502.20558v2)

缓存与分组也有信息边界：模拟器内部可以按真实轨迹加速生成量子样本；decoder 的模型选择或分组只能使用实际观测记录与公开协议。内部模拟分组 ID 不能暴露给 decoder。

**8. 先解决逻辑结果的定义，再接大规模 decoder。** 丢失 data 后，原有物理读出 parity 未必可直接计算。不能把 LOST 填成 0 当有效观测，也不能删除包含丢失位置的 logical observable、减少 k，或用每条 loss-conditioned circuit 的新 reference sample 偷换原始逻辑目标。

建议小码基准采用随机逻辑标签的双基记忆任务：分别在 Z 与 X 基准备均匀随机的 k 位逻辑标签 b，经过同一噪声过程，decoder 从 syndrome、终局读出与 herald 估计 b。准备标签及其参考值只由模拟/计分端保存；传给 decoder 的公共电路模板不包含秘密标签。标签通过噪声窗口外的无噪逻辑 Pauli 或等价 frame 随机化注入，不添加一组额外有噪物理门；需先验证初始化和逻辑算符约定。

零 loss 时，与现有错误帧评分的适配关系应逐 shot 验证：原始逻辑读出 parity `r = b XOR e`，现有 decoder 根据原 detector 输入 D 给出错误帧估计 `e_hat(D)`，则 `b_hat = r XOR e_hat(D)`，因此 `b_hat XOR b = e XOR e_hat(D)`。这一恒等式成立才保证原错误计分保持一致。若改用全部原始记录 m 或更强的 ML 后端，不再自动等价于原 telescoping；这种后端差异要单独比较，不能由零 loss 的物理通道等价推断评分也相同。

这一定义避免“已知全零输入，所以缺失时永远输出零”的伪恢复。它衡量两个共轭基的逻辑记忆读出可靠性，不应未经证明就等同于完整量子通道的 entanglement fidelity。若后续需要更强的量子存储结论，可另加与隔离逻辑参考系统的相关性检验。

为使普通错误率也有完整分母，约定每个正常完成的物理 shot 最终都有 k 位预测。decoder declared failure 时，由计分适配层使用独立 RNG 生成与真实标签无关的均匀 k 位猜测，同时保留失败 flag；该规则在所有 profile 上一致。基础设施异常或 backend unsupported 不伪装成这类物理预测，应单独报错或终止该评估。

对每个基报告：

- `logical_bit_error_rate`：全部 shots、全部 k 位上的平均错误比例，包含上述失败后的猜测。
- `logical_block_error_rate`：至少一个逻辑位错误的 shot 比例，包含上述失败后的猜测。
- `declared_failure_rate`：译码器主动报失败或未收敛的比例。
- `effective_block_failure_rate`：错误或 declared failure 的并集比例，所有物理 shots 保留在分母。
- 原子 loss 数、loss shots 比例、reload 数和实际执行时间等诊断量。

对 k 个标签完全无信息、且必须输出猜测的情况，块成功率应为 `2^-k`，逐位错误率应为 1/2。若协议允许并选择 declared failure，则按预先冻结的失败口径计分。不要把所有 loss 一律视为失败，也不要仅报告无 loss shots 的条件 LER。

数据丢失会随轮次累积，首版不一定满足平稳的逐轮错误假设。因此主要报告 T 轮块失败率与逐位率；若沿用项目既有归一化指标，同时报告原始指标，并注明其定义。不能直接除以 T 就称为测得的每轮逻辑错误概率。

与现有管道比较前，先证明零 loss 下上述任务与现有两基逻辑计分相容。不要直接将不同定义的数值放在同一排行榜。

**9. Decoder 的最小正确路线。** 首个 loss decoder 是小规模正确性参考，随后才实现可扩展近似。现有静态 noisy-circuit → detector/observable arrays → 固定 DEM/GARI 链路不能原样处理持续缺位。

完整的可观测条件化目标可写为：

\[
\hat b=\arg\max_b P(b\mid m,h),\qquad
P(b\mid m,h)\propto P(b)\sum_\tau P(\tau)P(m,h\mid b,\tau),
\]

其中 m 是实际有效的量子读出记录，h 是可见 loss/reload 信息，τ 是隐藏的 loss 历史；Pauli 及隐藏 trace 测量结果在似然内边缘化。标签先验为均匀分布。实现必须求逻辑类别的总概率，不能只选择概率最大的单个物理故障当成严格 ML。

第一步在少数原子的微型 Clifford 电路上，用密度矩阵或完整稳定子分支枚举实现这一基准。先验证单次 loss、多个可能时刻对应同一 herald、以及末端 erasure 的例子。只在规模确实可枚举时称“精确”；若截断故障阶数或 loss 数，报告截断尾概率及适用范围，不能将小码自动等同于可穷举整个 T=12 电路。

第二步才接 original-system telescoping 或其他适合一般 CSS 检查的 decoder：从可见 herald history 构造有效观测约束、逻辑映射与概率模型，并与参考结果比较。若使用加权合并不同 loss 假设的 DEM，需要明确这是近似；混合多个模型一般不会自动变成独立 Bernoulli 错误模型。多处损失的联合假设也不能无条件拆成独立的局部更新。

删除 CZ 后，原先确定的 detector 可能变成 gauge，最终 logical observable 也可能随机。需要重新构造有效约束或消除未观测量；不能仅维持原数组长度。对一般 qLDPC，不能照搬 surface-code 的图边拼接规则。保留逻辑维数 k 和不可观测的逻辑不确定性；不是把相关行列全删掉。

上游 telescoping 文档明确列出 GARI 对 detector 类型与 pure-component 结构的限制，heralded 测量结果不能直接作为 GARI detector 的引用；原系统路径可绕过这些特定结构限制。[上游接口说明](https://github.com/a7b/yarn/blob/main/telescoping-decoder/README.md)

但是 `use_gari=False` 不会自动解决 loss 的时间边缘化、gauge detector、随机 logical observable 或正确的逻辑计分。`allow_gauge_detectors` 也不是完整解决方案。实施时核对仓库锁定的上游版本；不要按远端 main 假定本地拥有同一 API。

建议将 decoder 能力分成三个明确状态：`reference_exact_small`、`validated_approximate_loss`、`unsupported`。在近似没有完成上述验证前，只开展小规模 loss 研究；不能静默丢弃 herald、转回原来的 GARI，再声称完成了 loss-aware 搜索。若 NA 和 SI1000 使用不同译码后端，还应增加双方同用兼容后端的对照，避免把 decoder 差异当成 noise 差异。

若评估遇到未支持的必要 loss/观测模式，整个评估应标记为不完整，保留已采样计数与未支持比例作诊断，不发布可比较的正式 LER。只有事先定义并验证的 decoder 拒绝/失败策略才能计入 declared failure；不能在看到 unsupported 后临时把它改成物理失败或删除该 shot。`decoded` 只表示算法返回预测，是否正确由私有参考计分，decoder 自身不知道。

**10. 代码接入地图。** 以下路径均相对于项目包目录，链接指向绝对位置。带“拟新增”的路径是建议，不是现有 API。尽量先保留抽象 SEC 结构，新增物理发射与 loss 执行层。

| 接入位置 | 现有入口或证据 | 实施内容 |
| --- | --- | --- |
| [config.py](omniqec_agentic/config.py) | `HardwareTarget`、`HARDWARE_PROMPT_PROFILES` 已核实 | 明确 ATOM 标签与评估 profile 的绑定；非法或未配置组合明确报错 |
| [verification/noise.py](omniqec_agentic/verification/noise.py) | `NoiseProfile`、`NoisePoint`、`config_hash`、`_canonical_dict` 已核实 | 增加两种 NA profile、参数校验、序列化与 provenance |
| [sec/stim_builder.py](omniqec_agentic/sec/stim_builder.py) | `_resolve_lowering`、`_Lowering` 已核实 | 接入 CZ-native lowering；发射 policy 固定，禁止 ATOM 回退 uniform |
| [sec/twobasis_builder.py](omniqec_agentic/sec/twobasis_builder.py) | 两基时间线由文档确认 | 保留两基 metadata；加入稳定的物理 event ID、读出 slot 与持续时间 |
| [verification/circuit.py](omniqec_agentic/verification/circuit.py) | `MemoryExperiment`、`MemoryExperimentPair`、`build_memory_experiment_pair` 由文档确认 | 区分静态 Pauli 电路与带操作流的 loss 实验；明确逻辑目标 |
| 拟新增 `verification/na_loss_model.py` | 新接口 | loss 概率、occupancy 更新、幸存端通道与 protocol 数据类 |
| 拟新增 `evaluation/na_loss_sampling.py` | 新接口 | 轨迹采样、trace、公开观测生成、私有 truth 隔离 |
| 拟新增 `verification/na_loss_reference.py` | 新接口 | 微型电路的 ML/枚举参考与逻辑计分 |
| [verification/decoders.py](omniqec_agentic/verification/decoders.py) | `build_observable_decoder`、`TelescopingObservableDecoder` 由文档确认 | 能力检查与后端分派；loss 条件模型可另建模块 |
| [evaluation/worker.py](omniqec_agentic/evaluation/worker.py) 与 [evaluation/sampling.py](omniqec_agentic/evaluation/sampling.py) | worker、采样与 shard 合并路径已确认 | 支持两种采样协议；记录所有失败类型与 loss 统计 |
| [evaluation/dem_cache.py](omniqec_agentic/evaluation/dem_cache.py) | 缓存职责由文档确认 | 静态 DEM 与 loss 条件模型分开缓存，完整检查 key |
| [workflow/state.py](omniqec_agentic/workflow/state.py) 与 [workflow/checkpoint.py](omniqec_agentic/workflow/checkpoint.py) | protocol snapshot 与 resume 检查已确认 | 新协议身份、禁止跨物理模型续跑，保留已有 uniform 协议 |
| [cli.py](omniqec_agentic/cli.py) 与 [export_stim.py](omniqec_agentic/export_stim.py) | 文件职责已确认 | NA 配置入口、帮助文本、导出 metadata；loss 导出不能假装单份 Stim 覆盖整个协议 |

配置传播还需定位 `NoiseSweepConfig`、`VerificationConfig` 及 legacy-p 转换路径；前次文档有引用，但本方案不假定其当前定义位置。实施时应搜索实际调用链再修改。

建议先将同一 lowered 物理操作流送入两种执行器：静态 Pauli 执行器生成 Stim 电路；loss 执行器逐 event 管理 occupancy 与观测。这能减少“两个后端其实执行不同理想电路”的风险。event ID 必须由确定的操作序列生成，不依赖 shot 内分支。

**11. 可复现、缓存与运行产物。** 每次评估的身份至少覆盖：候选码、SEC 模板与顺序、T、native lowering 版本、H 化简政策、全部噪声数值、loss 插入位置、检测与 reload 协议、idle 时间模型、采样器版本、decoder 类型与近似、逻辑计分定义、种子方案，以及依赖版本。

不改变调度时，抽象 SEC 缓存可以共享；改变门序、layer 时长或编译政策后，相关下游缓存必须失效。loss decoder 的条件缓存额外包含可见 herald/reload pattern 和条件模型版本；不能包含作为模型选择依据的隐藏真实 loss 时刻。大量不同 pattern 时采用有界缓存，报告命中率与构建时间。

随机性建议按 `base_seed / shot_id / event_id / atom_generation / random_kind` 分离，避免分支少抽一个随机数导致其他原子的所有事件漂移。trace、Pauli、loss、readout 和秘密逻辑标签使用独立流。同一批 shot 无论如何切 shard，应该保持结果一致；若底层量子采样器做不到逐位一致，需要在正式使用前明确固定分块政策，不能宣称已经获得分块无关复现。

结果除原有 LER 与资源数据外，至少导出：

```text
noise_config.json              # 数值、单位、来源、简化、hash
protocol_snapshot.json         # 调度、检测、reload、metric、依赖
compiled_operation_manifest   # event、targets、duration、measurement slots
score.json                     # 分基错误、declared failure、总分母、CI
loss_summary.json              # loss、暴露、reload、缓存与runtime统计
observed_records_sample       # 小批审计样本，不含私有真值
private_validation_truth      # 仅测试/审计可读，生产decoder不得访问
```

静态 `.stim` 可以表达 NA Pauli 基线，或某条已指定轨迹的条件电路。完整 persistent-loss 实验应导出操作流、配置与协议；一份普通静态 `.stim` 文件不能被宣称完整包含了所有动态 occupancy 与延迟译码语义。

`unsupported`、依赖缺失、非法配置、程序错误与正常 decoder 未收敛分别记录。程序错误不应伪装成某个候选物理性能差。是否将未收敛计为失败按协议冻结；不支持的候选需明确披露，不能默默从一个平台候选池删掉后声称公平比较。

**12. 实施顺序与每阶段验收。** 建议按以下顺序交付，每一步都能独立检查，不先启动完整 NA 搜索。

| 阶段 | 交付 | 进入下一阶段的条件 |
| --- | --- | --- |
| A：配置与 NA Pauli | 参数对象、profile identity、CZ-native 发射、两基 smoke | 零噪声理想行为正确；通道位置、H 计数、概率单位正确；无 loss 路径能使用既有评估链 |
| B：持续 loss 采样 | 操作流、occupancy、trace、LOST 读出、显式 reload | 手工单故障例子与解析 loss 分布一致；私有 truth 与公开记录隔离 |
| C：小规模译码与计分 | 随机逻辑标签任务、参考 ML、早/晚 loss 边缘化 | 可恢复与不可恢复 erasure 均计分正确；无 loss 对齐现有指标；无标签或时刻泄露 |
| D：可扩展 loss 后端 | 条件模型、近似验证、缓存、worker/shard 与协议闭合 | 相对参考结果的误差有记录；全部 shots 计入；不同配置不串缓存；性能可支持候选评估 |
| E：平台差异 pilot | 公共候选池、统一指标、交叉评估和消融 | 排名差异的统计与物理来源可解释，或明确报告尚无显著差异 |
| F：搜索闭环 | NA 评估真正进入候选选择与最终复测 | 同协议可复现；独立复测与搜索样本分离；产物可追溯 |

阶段 A 完成后可以先做操作噪声比例的 pilot；阶段 C 完成后再把 loss 纳入小码 pilot。阶段 D 是将 loss 纳入大规模搜索的前提。只用 NA Pauli 搜索、最后给赢家加 loss 复测，不应称作 loss-aware discovery。

建议的必要测试如下，测试目标是物理语义和接口边界，而不是照抄代码实现：

- 全零噪声：两基理想 detector/逻辑关系正确；lowering 不改变目标电路。
- ℓ=0：loss 执行器与 NA Pauli 执行器的可观测分布、逻辑计分一致。
- Bell 对丢失一个原子：幸存端约化态正确，隐藏 trace 测量结果不可见；dummy 不重新参与门。
- 单原子 m 个计划脉冲：生存率符合 `(1-ℓ)^m`；m=0、ℓ=0、ℓ=1 的边界正确。
- 一个伙伴已缺失：幸存端仍有 loss 风险，Pauli 边缘通道符合首版定义；两个都缺失不产生新损失。
- 同一 ancilla 的早/晚 loss：读出均为 LOST，但所影响的后续操作不同；普通 reset 不复活，显式 reload 才复活。
- 同一个公开 herald 对应多个隐藏 τ：decoder 不因 truth 文件或模拟分组不同而获得额外时刻信息。
- 终局 data loss：可恢复 erasure 不自动判失败；完全无逻辑信息的猜测基线正确；k 保持不变。
- 新增读出、LOST slot 与 reload：measurement record 索引和两基标签不串位。
- 声明失败、零失败及 postselection：总分母正确，零观测失败报告上置信界而非“真实 LER=0”。
- 分 shard 重放、配置变更与 resume：随机结果政策明确，缓存和协议隔离有效。
- SI1000 与 uniform 的受影响回归：已有协议的定义和输出未被 NA 参数污染。

在微型电路通过后，再用小 CSS 码的可控故障注入测试，然后选择四个 grammar family 中通过现有 SEC 认证的少量代表码做端到端资格检查。大码只用于后续吞吐与兼容性资格测试，不能替代小码物理正确性验证。

**13. 怎样检验“各有偏好”，而不是人为制造差异。** 首先固定一组跨 family、检查权重、连接度、k、X/Z 结构与深度都有差异的候选。可以从约 20–40 个候选起步，但数量不是验收门槛。所有 profile 测同一候选池，并采用相同的资源约束、逻辑任务和计分口径；同时报告 n、k、检查数量和电路成本，避免块错误率与编码量的混淆。

比较 uniform、SI1000、NA Pauli 与 NA+loss。分别做关闭 loss、改变测量/等待比例、必要时对 Pauli 分布对称化的消融；还要有仅改变整体噪声强度的对照。若做强度匹配，应使用同一个事先固定的参考电路进行一次标定，不能逐候选归一化其噪声暴露量。

独立搜索完成后，取两边候选的并集重新交叉测量，而不是只在各自模型里展示自己的赢家：

| 候选来源 | SI1000 下的独立复测 | NA 下的独立复测 |
| --- | --- | --- |
| SC 搜索候选 | LER、CI、资源 | LER、CI、资源 |
| NA 搜索候选 | LER、CI、资源 | LER、CI、资源 |

首先检查排名相关性、优胜集合重叠和成对差异置信区间；再检查差异是否在预先冻结的合理参数邻域中持续，以及能否由操作成本、loss 暴露/检测延迟等解释。仅整体 p 缩放也可能导致交叉，因此 crossing 本身不是平台特异性的充分证据。搜索和确认采用独立 shots；有搜索随机性时，用重复 seed 判断赢家变化是否稳定。

如果两边始终偏好相同或统计不可区分的候选，结果应写为“在当前码空间、SEC、decoder 与参数范围内未发现明确的平台特异性”。不要通过看到结果后增大 ℓ、指定强 Z bias 或挑选工作点来保证赢家相反。

若只改 noise 的 pilot 区分度有限，下一阶段再加入具体架构成本：选定超导连接图与路由政策，选定 NA 的移动或长程门实现，并把门数、等待、并行限制计入。SI1000 本身不定义连接图；抽象 all-to-all 也不等价于 NA 任意连接免费。长程中性原子 qLDPC 研究明确考虑了连接实现的错误代价，可作为扩展参考。[来源：High-rate quantum LDPC codes for long-range-connected neutral atom registers](https://www.nature.com/articles/s41467-025-56255-5)

**14. 后续扩展的触发条件。** 不要求首版同时完成这些工作：

- 若长期 memory 是研究目标，加入具体的 data loss 检测与恢复协议，并计入额外原子、门、误检、漏检与耗时。理想免费逐轮检测只能作为对照。
- 若使用实测相关 loss，两个在场原子的分布可显式写成 `P(none), P(a only), P(b only), P(both)`，概率归一；单端缺失时另定义脉冲风险。不能由单个边缘 ℓ 唯一推出 pair correlation。
- 若加入 Pauli bias，优先直接存储明确顺序的三项或十五项概率；只在有来源时给出硬件数值。若定义 `η=pZ/(pX+pY)`，各向同性的 η=1/2，不能错标为 1。实际 H 会交换 X/Z，CZ+H 实现的 CX 不能直接当 bias-preserving gate；当前搜索/电路中未实现的 H deformation 也不能被当成已有适应能力。[偏置与门实现的讨论](https://arxiv.org/html/2502.20558v2)
- 若读出过程中也有 loss，需指定它发生在状态信息取得之前还是之后，以及记录的是 bit、LOST 或两者；不能简单再加一个最终随机 mask 而不定义观测语义。
- 若从 Rb 切到 metastable Yb erasure conversion，建立单独 profile、物理状态和观测定义。真实 Rb 的完整故障容错架构则可参考分区、读出与补原子方案，但须逐项核实其参数，不能只摘取最优数字。[Rb 架构论文](https://www.nature.com/articles/s41586-025-09848-5)

**15. 实施时的完成清单。** 可按下面顺序勾选；其中数值来源冻结、loss 逻辑计分和译码验证是科学结果的必要部分。

- [ ] 新 profile、参数 schema、单位、来源与哈希闭合。
- [ ] NA Pauli 的 CZ-native 两基电路通过零噪声与受影响回归。
- [ ] occupancy、trace、读出 LOST 与显式 reload 的语义测试通过。
- [ ] decoder 输入不包含真实 loss 时刻、隐藏 trace 结果或秘密逻辑标签。
- [ ] 终局 data loss 的逻辑目标、k、分母和 declared-failure 口径固定。
- [ ] 小规模精确参考完成；可扩展近似的误差与限制已验证和记录。
- [ ] worker、shard、缓存、导出、protocol snapshot 与 resume 路径闭合。
- [ ] 公共候选池完成 NA Pauli、NA+loss 与 SI1000 的交叉评估及强度对照。
- [ ] 正式实验配置在最终确认前冻结，示例配置不冒充实测校准。
- [ ] 若开展 NA 搜索，loss-aware LER 真正进入选择闭环，最终结果用独立 shots 复测。

第一项可立即开始的开发工作是 `na_pauli_v1` 的配置与 CZ-native 发射；最先需要解决的科学接口是持续 loss 的观测与计分，而不是挑一个能让排名反转的 ℓ。

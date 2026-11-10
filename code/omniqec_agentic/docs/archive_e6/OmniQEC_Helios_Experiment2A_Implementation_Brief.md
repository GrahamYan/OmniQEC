# OmniQEC Experiment 2A：Helios 专属噪声适配与码搜索实现说明

**副标题：** 面向 98 物理比特预算的 calibration-informed、routing-independent 初版方案（含 modern SI1000 参考实现）  
**版本：** v0.5（2026-08-09，代码已实现）

> **一句话方案**：将 family program 编译为“CSS ancestor + 可选 sector-H local-Clifford deformation”，在相同 evaluator 下比较 `gb`、论文 `tb`、general `hgp`、`general_lp` 与 non-CSS `btlp`，并用 SI1000/Helios circuit-level LER 判断结构非对称和 Pauli-frame 非对称各自是否有效。

### 0.1 当前实现状态

本说明中的核心接口已经落入代码：

- `construction/tb.py`：实现 $z=xy$ 的论文 TB；`construction/two_seed_hgp.py` 与 `construction/two_seed_lp.py` 分别承载 canonical `hgp` 与 `general_lp` 的共享数学实现；旧名称只作为兼容别名；
- `construction/build.py`：新增 `build_compiled_code(spec)`；统一保存 CSS ancestor、canonical family、deformation、sector sizes 与 Hadamard mask。`general_lp`/`btlp` 共用同一 validator、lift 与矩阵构造，不复制 LP 数学代码；
- `verification/noise.py`：集中保存四个版本化 profile、参数来源、`p/scale` 轴、RZZ orientation 与配置哈希；
- `verification/circuit.py`：保留 uniform legacy builder，同时加入 BTLP mixed-stabilizer synthesis、显式 `TICK` 的 SI1000 CZ-native，以及 Helios CX/CZ/XCX native lowering/noise injection；
- `slow.py`、`verification/sinter_sweep.py`、`postsearch.py`：分别负责近似 DEM 搜索、noisy Stim circuit 最终采样、以及含 profile/矩阵哈希/Stim 版本的 artifact 导出；
- slow artifact schema 已升级为 v5；v4 默认迁移为 `deformation=none`，v3/旧 `p_values` 仍显式迁移为 `uniform_legacy_v1`。Experiment 2A 默认 family 为 `gb tb hgp general_lp btlp`。

当前自动化回归位于 `tests/test_experiment2a_*.py`。它覆盖论文 TB `[[30,4,5]]`、BTLP `[[12,2,3]]` fixtures、symplectic commutation、CX/CZ/XCX tableau 等价、mixed memory circuit、SI/Helios DEM、direct-circuit 与 DEM 边缘分布、schema/CLI 迁移以及 uniform legacy circuit/DEM golden hash。

### 0.2 v0.5 更新摘要

1. 将不准确的“two-seed 是新 family”术语修正为 general HGP/LP；两者本来就由两个独立 seeds 定义。代码使用 `hgp`、`general_lp`，旧 `two_seed_hgp/two_seed_lp` 仅供旧实验读取。
2. 新增 `btlp`：LLM 仍只搜索 $A/B/L$，deformation 固定为 `sector_h`，禁止生成任意 H-mask 或任意 non-CSS stabilizer。
3. 内部表示从裸 $H_X/H_Z$ 扩为 `CompiledCode`，而兼容接口 `build_matrices(spec)` 仍返回 CSS ancestor，避免破坏历史调用。
4. structural validation 对 BTLP 使用一般 symplectic commutation；fast loop 使用 local-Clifford-invariant ordinary distance，BTLP 不再伪报 non-CSS pure-axis $d_X/d_Z$。
5. syndrome synthesis 支持 mixed X/Z initialization、measurement 与 CX/CZ/XCX interaction；SI1000 最终仍全为 CZ，Helios 新增经过 tableau 验证的 CZ/XCX lowering。
6. CSV 保持 metrics-only；BTLP 的 `d_x_proxy/d_z_proxy` 留空，ancestor basis-resolved proxy 只进入 JSON artifact。deformation-aware hash 防止 `general_lp` 与相同 $A/B/L$ 的 `btlp` 被错误去重。
7. N≤40 预实验固定 SI1000 $p=10^{-3}$ 与 Helios `scale=1.0`，使用 `--n-budget 98 --proxy-n-max 40`。

## 0. 本文档的用途与边界

本文档用于让后续开发者或 AI 在不重新梳理讨论历史的情况下，快速实现 OmniQEC 的 Helios 专属子实验。

除 Helios 主实验外，本文档同时固定一个论文对齐的 **modern SI1000（Superconducting-Inspired 1000 ns）参考 evaluator**。SI1000 用于：

- 验证 shared “noiseless native circuit → noise injection → Stim DEM”基础设施；
- 与现有 uniform noise 区分，研究 measurement/reset-dominated circuit noise；
- 在需要时提供一个可复现的 superconducting-inspired 参考 campaign。

SI1000 **不是** H-native/H-sym 的第三个 Pauli-composition control。它使用不同的 native gate set、不同的时序噪声语义和不同的物理参数，结果必须单独报告。

**主实验包含：**

- 98 物理比特总预算；
- 对称 BB 作为 baseline；
- general HGP 与 general LP 作为可产生 X/Z 结构非对称的 CSS search families；
- BTLP 作为 fixed sector-H local-Clifford image，用于研究 Pauli-frame asymmetry；
- Helios native $R_{ZZ}(\pi/2)$ 的公开 15 分量 Pauli channel；
- 1Q、reset 和 measurement 的简化 Helios-1E 参数映射；
- Stim circuit 到 detector error model（DEM）的自动转换；
- BP-family decoder 使用保留 hyperedge 的 undecomposed DEM；
- top-$K$ 最终确认使用 noisy Stim circuit 的 exact detector sampling；
- 与总 Pauli fault probability 完全相同的 symmetrized control；
- cross-evaluation，用于证明发现的是 noise-adaptive code，而不是偶然的 campaign 差异。

**可选参考实验包含：**

- modern SI1000 的 CZ-native physical circuit；
- gate-dependent 1Q/2Q、reset、measurement、ordinary idle 与 resonator-idle noise；
- 由 Stim 从 noisy circuit 自动生成的 SI1000 DEM；
- 明确的“superconducting-inspired noise profile”claim boundary，不宣称具体超导芯片 calibration 或 routing fidelity。

**主实验暂不包含：**

- 离子逐步 routing、ring rotation、merge/split、cooling 的精确调度；
- “最多 4 个并行 2Q gate”导致的 wall-clock time；
- routing-duration-dependent memory noise；
- MCMR spectator crosstalk 的批次级精确建模；
- persistent leakage / seepage dynamics。

因此初版的准确表述应为：

> **Helios-calibration-informed circuit-level noise-aware code discovery**

而不应表述为：

> exact Helios digital twin、Helios routing-aware optimization，或完整 processor-specific compilation。

---

## 1. 与现有 OmniQEC 的衔接

OmniQEC 当前由 LLM 生成 family 内的 executable construction program。程序执行后得到具体 $H_X,H_Z$，fast loop 进行合法性、资源与 code-level 指标筛选；promoted candidates 进入 slow loop，由固定 syndrome-extraction synthesis 生成 Stim memory circuit，再注入噪声、构造 DEM、解码并计算 circuit-level LER。该子实验不改变这一接口，只扩展 family constructor，并替换 noise injector。[O1]

### 1.1 保持不变的部分

- Orchestrator、campaign memory 和反馈机制；
- construction program 作为 LLM 的输出；
- CSS ancestor 检查 $H_XH_Z^T=0$，BTLP 额外检查一般 symplectic commutation；
- 固定的一-check-one-ancilla syndrome extraction；
- Tanner graph edge coloring；
- detector、logical observable 与 decoder 接口；
- slow-loop 主排序指标：per-logical, per-round LER；
- 总资源约束：沿用当前 OmniQEC 的 $N_{\mathrm{phys}}\le 98$ accounting。

### 1.2 已完成的新增或修改

1. 新增论文 `tb`、canonical `hgp`、`general_lp` 与 `btlp` family；
2. characterization 中分别记录 $d_X,d_Z,w_X,w_Z,m_X,m_Z$；
3. 新增 `helios_calibration_informed_pauli_v1` noise injector；
4. 新增 `helios_symmetrized_v1` control injector；
5. 新增 `si1000_modern_v1` reference evaluator；
6. 将 profile-specific syndrome schedule 从压平的 `cx_order` 改为保留 `TICK` 的 parallel layers；
7. 增加 Helios channel operand-orientation sensitivity；
8. Sinter 最终路径从 noisy circuit 直接采样；native/sym top-$K$ cross-evaluation 由相同 post-search 接口分别执行；
9. slow artifact schema 升级至 v5，保存 deformation、sector sizes、mask/symplectic/CSS-ancestor hashes；
10. metrics-only CSV 直接汇总 family、LER、shots、failures、置信区间和 distance 指标。

---

## 2. 科学问题与实验设计

### 2.1 核心问题

在相同的 98 物理比特预算、相同 syndrome synthesis 和相同 decoder 下：

> Helios 原生 gate channel 中的结构化 Pauli asymmetry，是否会使 OmniQEC 从对称 BB/TB、结构非对称的 HGP/LP，以及 Pauli-frame 非对称的 BTLP 中，发现不同的最优 construction program，并获得更低的 circuit-level LER？

这里的重点不是预设“非对称 code 一定更好”，而是检验 slow-loop evidence 是否会改变 family selection 和 family parameters。

### 2.2 两个主 campaign

#### Campaign H-native

- 2Q gate：Helios native $R_{ZZ}(\pi/2)$ 的公开 15 分量 Pauli channel；
- 1Q：简化 Pauli fault；
- reset / measurement：Helios-1E emulator split；
- routing memory、MCMR crosstalk、leakage：关闭。

#### Campaign H-sym

- 与 H-native 使用完全相同的总 2Q non-identity Pauli probability；
- 将该概率平均分配到 15 个非平凡双比特 Pauli 项；
- 其他 1Q、reset、measurement 设置保持相同。

这使两组实验只在 **Pauli composition** 上不同，而不是总 error strength 不同。

### 2.3 必须进行的 cross-evaluation

至少保存两组 campaign 的 top-$K$ codes，并形成以下矩阵：

| 搜索来源 | 在 H-native 下评估 | 在 H-sym 下评估 |
|---|---:|---:|
| H-native top-$K$ | 必须 | 必须 |
| H-sym top-$K$ | 必须 | 必须 |

有说服力的结果应表现为：H-native 搜出的 codes 在 H-native 下系统性优于 H-sym 搜出的 codes；而该优势在 H-sym 下缩小、消失或反转。

### 2.4 主指标与辅助指标

**主指标：**

- circuit-level per-logical, per-round LER。

**辅助解释指标：**

- family；
- $[[n,k,d]]$ 与 basis-resolved $d_X,d_Z$；
- $w_X,w_Z$；
- $m_X,m_Z$；
- data-qubit degree distribution；
- syndrome-extraction depth；
- 2Q gate count；
- H-native / H-sym cross-evaluation gap。

---

## 3. 搜索 family

### 3.0 统一内部表示

所有 family spec 先通过 `build_compiled_code(spec)` 编译为：

```text
CompiledCode(
    hx, hz,                       # CSS ancestor
    family,                       # canonical family name
    deformation,                  # none | sector_h
    h_mask,                       # data-qubit Hadamard mask
    sector_sizes,
    code_hash                     # deformation-aware
)
```

`build_matrices(spec)` 继续返回 CSS ancestor $H_X,H_Z$，从而保持旧调用和普通 CSS family 的历史 hash 不变。只有 `btlp` 的 compiled hash 额外包含 Hadamard mask；其 symplectic stabilizer、mask 与 ancestor 另存独立 hashes，供 archive/cache/provenance 使用。

### 3.1 对称 baseline：BB

保留现有 BB constructor，作为：

- 对称 construction baseline；
- 当前 OmniQEC 已证明可有效搜索的 family；
- 检验 LLM 是否会在 H-native 下主动从熟悉的 BB frontier 转向新 family 的参照。

BB 不应从 H-native campaign 中删除，否则无法证明 family selection 是由 circuit-level evidence 驱动的。

### 3.2 论文 Trivariate Bicycle（TB）

TB 使用两个独立循环维度，第三个变量满足 $z=xy$：

$$x=S_\ell\otimes I_m,\qquad y=I_\ell\otimes S_m,\qquad z=S_\ell\otimes S_m.$$

它与三个独立循环维度、数据比特数为 $2\ell mr$ 的现有 `gb3` 不同。Experiment 2A 的 golden fixture 为 $\ell=3,m=5$、$A=x+z^4$、$B=x+y^2+z^2$，产生 `[[30,4,5]]`、row weight 5；总预算为 $4\ell m\le98$。[C3]

### 3.3 General HGP（canonical family：`hgp`）

令两个独立 classical parity-check matrices 为

$H_1\in\mathbb F_2^{r_1\times n_1}$，$H_2\in\mathbb F_2^{r_2\times n_2}$。

构造：

$H_X=[H_1\otimes I_{n_2}\mid I_{r_1}\otimes H_2^T]$，

$H_Z=[I_{n_1}\otimes H_2\mid H_1^T\otimes I_{r_2}]$。[C1]

若两个 classical matrices 分别具有 row/column weights $(\rho_1,\gamma_1)$ 和 $(\rho_2,\gamma_2)$，则 regular 情况下：

$w_X=\rho_1+\gamma_2$，$w_Z=\rho_2+\gamma_1$。

标准 HGP 从定义上就使用两个独立 classical seeds，并不要求 $H_1=H_2$。因此更准确的名称是 general HGP；LLM 独立搜索 $H_1,H_2$ 后，可产生 $w_X\ne w_Z$、$d_X\ne d_Z$ 或不同的 X/Z redundancy，而 CSS commutation 由模板结构保证。

**建议 program 参数：**

```json
{
  "family": "hgp",
  "H1": [[1, 1, 0], [0, 1, 1]],
  "H2": [[1, 0, 1], [1, 1, 0]]
}
```

`two_seed_hgp` 保留为旧 archive/spec 的兼容别名，读取时自动规范化为 `hgp`，但不再出现在新 prompt 或默认 family 列表中。两个 seeds 必须真正独立，而不是同一个 seed 与其 transpose。

### 3.4 General LP（canonical family：`general_lp`）

令 $A\in R^{m_A\times n_A}$、$B\in R^{m_B\times n_B}$，其中 $R=\mathbb F_2[\mathbb Z_L]$ 或当前 LP implementation 使用的 group algebra。用 $^*$ 表示 protograph transpose 加 group inverse/conjugation。构造可写为：

$H_X=[A\otimes I_{n_B}\mid I_{m_A}\otimes B^*]$，

$H_Z=[I_{n_A}\otimes B\mid A^*\otimes I_{m_B}]$，

随后将 group-algebra entries binary lift。[C2]

General LP 的 search variables 包括：

- $A,B$ 的独立 shape；
- 每个 entry 的 monomial support；
- lift size $L$；
- row/column weight profile；
- cyclic group 或已支持的 group；
- random seed 与 program-level generation rule。

HGP 可视为不带非平凡 lift 的更简单 product baseline；LP 提供更大的有限尺寸 program-search 空间。标准 LP 本身就从两个独立 protographs 开始，因此 `two_seed_lp` 仅是旧名称兼容别名，canonical schema 为：

```json
{
  "family": "general_lp",
  "A": [[[0, 2]]],
  "B": [[[0, 1]]],
  "L": 6
}
```

### 3.5 Bias-tailored lifted-product（`btlp`）

BTLP 与 `general_lp` 使用完全相同的 $A/B/L$ validator、binary lift 和 CSS ancestor 构造。其唯一额外选择是固定的 family-level local-Clifford deformation：对第二 qubit sector 施加 Hadamard。LLM 只搜索 $A/B/L$，不能输出任意 qubit mask、任意 Clifford 或任意 non-CSS stabilizer：

```json
{
  "family": "btlp",
  "A": [[[0, 2]]],
  "B": [[[0, 1]]],
  "L": 6
}
```

若 $A\in R_L^{m_A\times n_A}$、$B\in R_L^{m_B\times n_B}$，两个 data sectors 的大小固定为

$$n_{\mathrm{sector1}}=n_A n_B L,\qquad n_{\mathrm{sector2}}=m_A m_B L.$$

前 $n_{\mathrm{sector1}}$ 个 data qubits 不变，后 $n_{\mathrm{sector2}}$ 个 qubits 应用 Hadamard。将 ancestor 分块为

$$H_X=[H_{X,1}\mid H_{X,2}],\qquad H_Z=[H_{Z,1}\mid H_{Z,2}],$$

则当前 check-row 顺序下的 symplectic stabilizer 为

$$
S_X=
\begin{bmatrix}
H_{X,1}&0\\
0&H_{Z,2}
\end{bmatrix},\qquad
S_Z=
\begin{bmatrix}
0&H_{X,2}\\
H_{Z,1}&0
\end{bmatrix}.
$$

验证条件不再是对 deformed code 复用 CSS 检查，而是

$$S_XS_Z^T+S_ZS_X^T=0,\qquad k=n-\operatorname{rank}[S_X\mid S_Z].$$

同时必须验证该 $k$ 与 CSS ancestor 一致。local Clifford 保持 Tanner support、check Pauli weight、普通 Pauli distance 和 $[[n,k,d]]$，但不会保持“纯 X/纯 Z 误差”的含义。因此：

- fast loop 继续使用 ancestor ordinary-distance proxy 作为 `d_proxy`；
- BTLP CSV 的 `d_x_proxy/d_z_proxy` 留空；
- JSON artifact 使用 `ancestor_d_x_proxy/ancestor_d_z_proxy` 明确保存祖先指标；
- exact distance 在 ancestor 上认证并标记 `local_clifford_inherited`；
- deformation-aware `code_hash` 包含 Hadamard mask，避免与相同 $A/B/L$ 的 `general_lp` 去重冲突。

论文/官方代码的 golden fixture $L=6$、$A=1+x^2$、$B=1+x$ 产生 `[[12,2,3]]`，作为实现回归测试。[C4,C5]

### 3.6 Fast-loop 约束建议

所有 family 使用共同的 hard constraints：

- CSS ancestor validity；BTLP 另做 symplectic validity；
- $N_{\mathrm{phys}}\le98$；
- $k>0$；
- 建议初版 $w_{\max}\le 8$，之后根据候选数调整；
- parity-check matrix 不含全零 row/column；
- 无明显重复 checks 或先做 rank reduction；
- circuit synthesis 成功；
- DEM 生成成功。

fast proxy 可继续沿用现有 $kd^2/n$。普通 CSS family 同时把 $d_X,d_Z$、$w_X,w_Z$ 提供给 orchestrator；BTLP 只暴露 ordinary `d_proxy`，ancestor basis-resolved 值放入带语义标签的 artifact。不要在 fast loop 中手工规定“必须非对称”，否则会削弱“自主选择 family/structure”的结论。

---

## 4. Helios 公开参数与来源

### 4.1 架构事实：本版本如何使用

Helios 是 98-qubit QCCD processor，提供 transport-enabled all-to-all connectivity；logic region 一批最多处理 16 个 qubits，官方 emulator 标注最多 4 个 parallel 2Q operations。[H1–H3]

**本实验只使用：**

- 98 physical-qubit budget；
- native gate set 与 calibration-informed noise parameters。

**本实验不使用：**

- 4-gate parallel limit；
- 7 gating rounds；
- 55 ms dense random layer；
- transport schedule。

原因是这些量在当前 Stim 实现中只有在显式加入 duration/transport-dependent noise 后才会改变 DEM；否则只改变 wall-clock time。

### 4.2 Native $R_{ZZ}(\pi/2)$ Pauli channel

Helios paper 的 2Q cycle benchmarking 给出 zone-averaged Pauli probabilities。其值包含对不可学习自由度的 symmetry assumptions，并且不包含 persistent leakage dynamics。[H1]

Stim `PAULI_CHANNEL_2` 的参数顺序为：

`IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ`。[S1]

| Pauli | Probability |
|---|---:|
| IX | $4.5\times10^{-5}$ |
| IY | $4.5\times10^{-5}$ |
| IZ | $19\times10^{-5}$ |
| XI | $5.8\times10^{-5}$ |
| XX | $0.06\times10^{-5}$ |
| XY | $0.06\times10^{-5}$ |
| XZ | $5.8\times10^{-5}$ |
| YI | $5.8\times10^{-5}$ |
| YX | $0.06\times10^{-5}$ |
| YY | $0.06\times10^{-5}$ |
| YZ | $5.8\times10^{-5}$ |
| ZI | $19\times10^{-5}$ |
| ZX | $4.5\times10^{-5}$ |
| ZY | $4.5\times10^{-5}$ |
| ZZ | $5.9\times10^{-5}$ |

总 non-identity probability：

$P_{2Q}=8.534\times10^{-4}$。

该值是 **computational-subspace Pauli fault probability**，不是包含 leakage 的完整 2Q average infidelity。对两比特 Pauli channel，$d=4$，因此其 computational-subspace average infidelity 为：

$$
\epsilon_{\mathrm{Pauli}}
=\frac{d}{d+1}P_{2Q}
=\frac45\times8.534\times10^{-4}
=6.8272\times10^{-4}.
$$

Helios paper 的 2QCB 同时报告约 $1.14\times10^{-4}$ 的 leakage rate；二者相加为约 $7.97\times10^{-4}$，与包含 leakage 的 $8.1(2)\times10^{-4}$ average infidelity 一致。[H1] 因此初版必须：

- 用上表替换而不是叠加 Helios-1E 的标量 `p2`；
- 明确把 persistent leakage dynamics 关闭；
- 不把该 15 项 channel 宣称为完整 Helios 2Q gate process。

该 channel 具有显著 Z/phase bias：

- `IZ + ZI` 占全部 non-identity Pauli fault mass 的约 $44.5\%$；
- `IZ + ZI + ZZ` 占约 $51.4\%$；
- 若把 `X/Y` 计为 bit component、`Z/Y` 计为 phase component，则第一操作数的 phase/bit marginal ratio 约为 $1.95$，第二操作数约为 $2.50$；
- 完全对称化 control 的相应 marginal ratio 为 $1$。

对称化 control 中，每一项取：

$P_{2Q}/15=5.689333\times10^{-5}$。

### 4.3 其他 Helios-1E 参数

以下为官方 Helios-1E emulator profile（标注日期 2025-11-18）的公开参数。[H3]

| Parameter | Default | 初版 Stim 映射 |
|---|---:|---|
| `p1` | $2.5\times10^{-5}$ | physical 1Q Clifford/Rxy 后的简化 `DEPOLARIZE1` |
| `p2` | $8.0\times10^{-4}$ | 不直接使用；2Q 使用上表 15 分量 channel |
| `p_init` | $5.0\times10^{-4}$ | basis-aware reset flip |
| `p_meas(0)` | $1.0\times10^{-6}$ | basis-aware measurement flip |
| `p_meas(1)` | $1.0\times10^{-6}$ | 同上 |
| `p_crosstalk_meas` | $5.5\times10^{-5}$ | 初版关闭 |
| `p_crosstalk_init` | $3.5\times10^{-6}$ | 初版关闭 |
| linear dephasing rate | $0.0079\;\mathrm{s}^{-1}$ | 初版关闭 |
| quadratic dephasing rate | $0.058$ | 初版关闭 |
| coherent-to-incoherent factor | $1.5$ | 初版关闭 |

注意：paper 报告的 standard SPAM 平均 error 为 $4.8(6)\times10^{-4}$；`p_init` 与 `p_meas` 的拆分来自 emulator model，而不是 paper 对 preparation 和 readout 的独立直接测量。[H1,H3]

`DEPOLARIZE1(p1)` 同样是 reduced-order approximation。官方 emulator 的 1Q/2Q physical model 还包含 asymmetric depolarization、spontaneous emission、leakage 和 duration-dependent dephasing；初版只保留一个对称 Pauli surrogate，不能称为对 Helios-1E 内部 noise engine 的逐项复刻。[H4]

---

## 5. 从 Helios 参数到 Stim circuit 与 DEM

### 5.1 原则：不要手写 DEM

正确流程是：

1. 生成带 `DETECTOR` 和 `OBSERVABLE_INCLUDE` 的 noiseless Stim circuit；
2. 在具体 gate/reset/measurement location 注入 Helios channels；
3. 调用 `stim.Circuit.detector_error_model(...)` 自动得到 code-specific DEM；
4. 把 DEM 与 detection-event samples 交给现有 decoder。

DEM 中每条 `error(p) D... L...` 的 detector targets 取决于具体 code、gate ordering 和 detector definitions，不能从 calibration table 预先手写。

### 5.2 必须先关闭现有四参数统一噪声

若当前 generator 使用：

- `after_clifford_depolarization`；
- `after_reset_flip_probability`；
- `before_round_data_depolarization`；
- `before_measure_flip_probability`；

则在生成 noiseless base circuit 时全部设为 0，再由 Helios injector 注入，避免 double counting。

### 5.3 对所有抽象 Clifford 做完整 Helios native lowering

不能只替换 `CX`。当前 memory circuits 中独立出现的 `H`、`RX` 和 `MX` 也必须降到 Helios native $R_{xy}$ / virtual $R_Z$ gate set，否则会漏掉 physical 1Q pulses 及相应的 `p1`。

Quantinuum 官方 CX decomposition 可在 Stim 中写成以下等价 Clifford sequence（按执行顺序）：

```text
SQRT_Y_DAG target
SQRT_ZZ control target
S_DAG control
SQRT_X_DAG target
S_DAG target
```

对应完整注噪位置为：

```text
SQRT_Y_DAG target
DEPOLARIZE1(p1) target

SQRT_ZZ control target
PAULI_CHANNEL_2(pIX, ..., pZZ) control target

S_DAG control                  # virtual Rz: no added noise
SQRT_X_DAG target
DEPOLARIZE1(p1) target
S_DAG target                   # virtual Rz: no added noise
```

其中：

- `SQRT_ZZ` 与 $R_{ZZ}(\pi/2)$ 在 global phase 下等价；
- 两个 `S_DAG` 是 virtual $R_Z(-\pi/2)$，初版不加 physical noise；
- 两个 $\pi/2$ `Rxy` physical rotations 后各加入一次 1Q noise；
- `SQRT_ZZ` 后加入一次 15 分量 `PAULI_CHANNEL_2`；
- 论文表在其 symmetry classes 下对 $R_{ZZ}(\pi/2)$ conjugation 不变，因此把 Pauli channel 固定放在 ideal `SQRT_ZZ` 后是明确且可复现的 convention。[H1,H2]

Helios-native Hadamard 为：

```text
SQRT_Y_DAG q
DEPOLARIZE1(p1) q
Z q                            # virtual Rz(pi): no added noise
```

该序列与 `H` 在 global phase 下等价。于是 `RX` 必须实现为 Z-basis reset 后再执行 native `H`，`MX` 必须实现为 native `H` 后再执行 Z-basis measurement，而不能直接保留一个没有 physical 1Q noise 的抽象 `RX`/`MX`。

推荐的核心实现骨架：

```python
from dataclasses import dataclass
from typing import Literal
import stim

HELIOS_RZZ_PAULI = [
    4.5e-5, 4.5e-5, 19e-5,                 # IX, IY, IZ
    5.8e-5, 0.06e-5, 0.06e-5, 5.8e-5,     # XI, XX, XY, XZ
    5.8e-5, 0.06e-5, 0.06e-5, 5.8e-5,     # YI, YX, YY, YZ
    19e-5, 4.5e-5, 4.5e-5, 5.9e-5,        # ZI, ZX, ZY, ZZ
]
HELIOS_RZZ_ORDER = [
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
]


@dataclass(frozen=True)
class HeliosNoiseConfig:
    p1: float = 2.5e-5
    p_init: float = 5.0e-4
    p_meas: float = 1.0e-6
    scale: float = 1.0
    symmetrize_rzz: bool = False
    rzz_operand_orientation: Literal[
        "control_first", "target_first", "swap_average"
    ] = "control_first"


def rzz_probs(cfg: HeliosNoiseConfig) -> list[float]:
    raw = dict(zip(HELIOS_RZZ_ORDER, HELIOS_RZZ_PAULI, strict=True))
    if cfg.symmetrize_rzz:
        p = [sum(raw.values()) / 15] * 15
    elif cfg.rzz_operand_orientation == "swap_average":
        p = [(raw[label] + raw[label[::-1]]) / 2 for label in HELIOS_RZZ_ORDER]
    else:
        p = [raw[label] for label in HELIOS_RZZ_ORDER]
    p = [cfg.scale * x for x in p]
    if sum(p) >= 1:
        raise ValueError("Invalid two-qubit Pauli channel.")
    if max(p, default=0.0) > 1e-3:
        raise ValueError("RZZ component exceeds configured DEM approximation threshold.")
    return p


def append_1q_noise(c: stim.Circuit, q: int, cfg: HeliosNoiseConfig) -> None:
    p = cfg.scale * cfg.p1
    if p:
        c.append("DEPOLARIZE1", [q], p)


def append_helios_h(c: stim.Circuit, q: int, cfg: HeliosNoiseConfig) -> None:
    c.append("SQRT_Y_DAG", [q])
    append_1q_noise(c, q, cfg)
    c.append("Z", [q])  # virtual Rz(pi)


def append_helios_cx(c, control, target, cfg) -> None:
    c.append("SQRT_Y_DAG", [target])
    append_1q_noise(c, target, cfg)

    c.append("SQRT_ZZ", [control, target])
    pair = [control, target]
    if cfg.rzz_operand_orientation == "target_first":
        pair = [target, control]
    c.append("PAULI_CHANNEL_2", pair, rzz_probs(cfg))

    c.append("S_DAG", [control])
    c.append("SQRT_X_DAG", [target])
    append_1q_noise(c, target, cfg)
    c.append("S_DAG", [target])
```

实现中建议用带名称的 Pauli-probability mapping 生成最终 15-tuple，并对 `target_first`/`swap_average` 做显式 Pauli-label transform，避免靠手写下标重排。

### 5.4 Initialization、measurement 与轮间 reset 语义

物理基元只使用 Z-basis initialization 和 measurement：

```python
def append_reset_z(c, q, cfg):
    c.append("R", [q])
    c.append("X_ERROR", [q], cfg.scale * cfg.p_init)


def append_reset_x(c, q, cfg):
    append_reset_z(c, q, cfg)
    append_helios_h(c, q, cfg)


def append_measure_z(c, q, cfg):
    c.append("X_ERROR", [q], cfg.scale * cfg.p_meas)
    c.append("M", [q])


def append_measure_x(c, q, cfg):
    append_helios_h(c, q, cfg)
    append_measure_z(c, q, cfg)
```

这里 `p_init`/`p_meas` 是 Helios-1E emulator split 的 Pauli surrogate；特别是 initialization leakage 没有被模拟成 persistent leakage。[H3,H4]

当前 circuit 每轮开头写 `R ancilla`、末尾又写 `MR ancilla`。在加入 physical reset noise 后，直接保留这种结构会在相邻 syndrome rounds 之间形成重复 reset，或者使前一个 `MR` 后的 initialization fault 被下一个 `R` 立即擦除。必须固定为以下两种等价策略之一：

1. 每轮末尾使用 `M`，下一轮开头恰好执行一次 `R + p_init`；或
2. 非最后一轮末尾使用 `p_meas + MR + p_init`，并删除下一轮开头的 `R`；最后一轮只使用 `p_meas + M`。

推荐第 2 种，因为 measurement record bookkeeping 与现有 `MR` 结构更接近。验收条件是：首次 ancilla preparation 有一次 initialization fault；任意两个相邻 syndrome rounds 之间恰好有一次 initialization fault；最后一次 measurement 后不注入不会再被使用的 reset fault。

### 5.5 Canonical BP DEM 与 Stim disjoint approximation

`PAULI_CHANNEL_2` 是一个 16-outcome categorical channel：15 个非平凡 Pauli outcomes 彼此互斥，剩余概率对应 `II`。Stim circuit sampling 会精确执行这个互斥分布；但标准 DEM 只能表示彼此独立的 binary error mechanisms，因此 circuit-to-DEM 转换需要显式允许 small-probability independent approximation。[S1,SI6]

当前 BP/BP-OSD 接口把每条 DEM `error(...)` 当作 $H$ 和 $L$ 的一列。canonical BP DEM 必须保留一次物理 fault 造成的完整 detector hyperedge：

```python
dem = noisy_circuit.detector_error_model(
    decompose_errors=False,
    flatten_loops=True,
    allow_gauge_detectors=False,
    approximate_disjoint_errors=1e-3,
)
```

不能在 canonical BP path 中使用 `decompose_errors=True`。该选项会输出带 `^` separator 的 graphlike decomposition；当前 `verification/dem.py::dem_to_matrices` 会忽略 separator，并不能保留其 component semantics。若未来为 PyMatching 生成 graphlike artifact，应使用独立的 decoder-specific DEM，并由理解 separator 的 parser 消费：

```python
matching_dem = noisy_circuit.detector_error_model(
    decompose_errors=True,
    flatten_loops=True,
    allow_gauge_detectors=False,
    approximate_disjoint_errors=1e-3,
)
```

名义模型中最大单项 probability 为 $1.9\times10^{-4}$，低于 $10^{-3}$ threshold。threshold 限制的是被近似的最大 **单项 disjoint probability**，不是 15 项之和。[SI6]

在 scale $=1$ 时，把 15 项互斥 outcomes 当成独立 mechanisms 后，单个 RZZ location 同时触发两个以上 mechanisms 的概率约为：

- H-native：$3.15\times10^{-7}$；
- H-sym：$3.40\times10^{-7}$。

它们分别约占该 location 总 Pauli fault mass 的 $0.037\%$ 和 $0.040\%$。虽然是高阶小量，但 experiment artifacts 必须记录 approximation threshold 和 Stim version，最终 LER 不能只依赖 approximate DEM sampling。

### 5.6 双路径评估：DEM 建 decoder，noisy circuit 做最终 exact sampling

推荐使用同一个 noisy Helios Stim circuit 产生两条评估路径：

```text
noisy Helios Stim circuit
    ├── detector_error_model(...)
    │      -> approximate independent DEM
    │      -> H / L / priors
    │      -> BP / BP-OSD decoder
    │
    └── compile_detector_sampler()
           -> exact categorical PAULI_CHANNEL_2 sampling
           -> detection events + logical observables
           -> decoder failure statistics
```

执行政策：

- search slow loop 可以继续从 approximate DEM mechanisms 采样，以控制成本；
- top-$K$ cross-evaluation 必须在固定 `scale=1` 下从 `noisy_circuit.compile_detector_sampler()` 直接采样 detectors 与 observables；
- decoder 的 $H/L$/priors 仍来自 canonical approximate DEM；因此 sampling distribution 是 exact circuit distribution，decoder prior 仍是独立机制近似；
- 保存 direct-circuit LER 与 DEM-sampled LER，并检查差异是否落在置信区间内；最终主结果以 direct-circuit LER 为准；
- 若两种 sampling paths 改变候选排序，必须增加 direct shots，并在结果中报告该差异，而不能静默选择较有利结果。

若未来希望连 decoder prior 也精确表达互斥 16-outcome variable，需要实现 categorical factor node 或相应的 non-binary factor-graph decoder；标准 Stim DEM 本身不能表达这种 exclusivity。

### 5.7 在 synthesis 阶段逐 layer lowering/injection

最稳妥实现是让 syndrome synthesizer 从 shared noiseless schedule IR 直接生成 Helios-native circuit，而不是先生成含抽象 `CX/RX/MX` 的最终 Stim circuit 再做脆弱的字符串 post-pass。这样可以：

- 对所有 `H/RX/MX/CX/CZ/XCX` 完整 lowering；
- 保留 CX control/target 以及 RZZ channel operand orientation；
- 对每个 edge-color layer 的并行 1Q、RZZ、virtual-Rz submoments 写入明确 `TICK`；
- 避免递归改写 `REPEAT` block；
- 避免误处理 measurement-record feedback target；
- 保留 detector 与 observable indexing；
- 统计 expected/actual physical `Rxy`、`RZZ`、initialization 和 measurement noise-location counts。

若必须 post-process，需递归处理 repeat blocks，并原样保留 `TICK`、`DETECTOR`、`OBSERVABLE_INCLUDE`、`SHIFT_COORDS` 等非门指令。即使 V1 关闭 duration-dependent noise，也必须保留层结构和 circuit hash，避免未来加入 transport/idle noise 时重新解释旧结果。

### 5.8 RZZ operand orientation sensitivity

论文表中的第一、第二个 Pauli 字符对应有序的物理 qubit pair；但 ideal $R_{ZZ}$ 是对称门，抽象 CX 的 logical control/target 并不天然等于物理离子的左右位置。当前 routing-independent model 没有足够信息恢复真实 ion placement，因此必须把映射 convention 显式版本化。

主 campaign 固定：

```text
rzz_operand_orientation = control_first
```

即 `PAULI_CHANNEL_2` 的第一个字符作用于 abstract CX control，第二个字符作用于 target。随后对 top-$K$ 至少执行：

- `target_first`：交换 channel 的两个 Pauli operands；
- 可选 `swap_average`：对每一项取 $[P(PQ)+P(QP)]/2$，消除未建模的 physical ordering。

raw channel 与完全 swapped channel 的 total-variation distance 为 $5.2\times10^{-5}$，约占 non-identity Pauli mass 的 $6.1\%$；raw channel 与 fully symmetric control 的 total-variation distance 约占 $32.0\%$。因此 orientation ambiguity 小于本实验研究的 native-vs-sym 差异，但仍可能影响有限尺寸 code ranking。

若 top-$K$ 排名对 `control_first`/`target_first` 稳定，可保留 raw `control_first` 为主结果；若排名明显变化，则必须把 `swap_average` 升为主模型，或进入包含 physical placement/routing 的 V2，不能把任意 control/target convention 解释为真实硬件优势。

### 5.9 BTLP mixed-stabilizer syndrome circuit

BTLP 的 Tanner support 与 CSS ancestor 相同，因此继续使用同一 edge coloring 和 one-check-one-ancilla schedule，不重新 routing。区别只发生在 sector 2 的 Pauli orientation：

| Memory experiment | Sector 1 interaction | Sector 2 interaction |
|---|---|---|
| inherited X-memory | `CX(ancilla,data)` | `CZ(ancilla,data)` |
| inherited Z-memory | `CX(data,ancilla)` | `XCX(data,ancilla)` |

- inherited X-memory：sector 1 data 使用 X-basis initialization/final measurement，sector 2 使用 Z basis；
- inherited Z-memory：sector 1 使用 Z basis，sector 2 使用 X basis；
- detector 与 observable support 沿用 ancestor，但 final measurement-record map 按每个 data qubit 的实际 basis 构造；
- decoder 不读取“CSS/non-CSS”标签，只消费实际 noisy circuit 产生的 DEM detectors/observables。

Helios 对三种 abstract two-qubit Clifford 使用以下按执行顺序的 lowering：

```text
CX(c,t):
  SQRT_Y_DAG(t)
  SQRT_ZZ(c,t)
  S_DAG(c)
  SQRT_X_DAG(t)
  S_DAG(t)

CZ(c,t):
  SQRT_ZZ(c,t)
  S_DAG(c)
  S_DAG(t)

XCX(c,t):
  SQRT_Y_DAG(c)
  SQRT_Y_DAG(t)
  SQRT_ZZ(c,t)
  SQRT_X_DAG(c)
  S_DAG(c)
  SQRT_X_DAG(t)
  S_DAG(t)
```

三者均已用 Stim tableau 做 ideal-Clifford 等价测试。每个 physical `Rxy` 后注入一次现有 `p1` channel，每个 `SQRT_ZZ` 后注入一次 15 项 RZZ Pauli channel，virtual `S/S_DAG/Z` 不注噪；因此 CZ 额外 Rxy 数为 0，XCX 为 4。RZZ channel 的 operand order 始终继承 ancestor edge 的 `(control,target)`，不因 CZ/XCX 的视觉对称性而交换。

SI1000 lowering 的最终 2Q gate仍全部是 CZ。inherited X-memory 的 syndrome block 在 ancilla 与 sector 1 data 上放 H frame；inherited Z-memory 则在 ancilla 与 sector 2 data 上放 H frame。同一 block 内允许消除可合并的 H，但不得跨 measurement/reset/round boundary。`h_mask=0` 的普通 CSS 路径必须保持历史 circuit/DEM hash。

---

## 6. Superconducting-Inspired SI1000 参考噪声与 DEM 实现

### 6.1 定位：它不是 Helios symmetrized control

SI1000（Superconducting-Inspired 1000 ns）来自 *A Fault-Tolerant Honeycomb Memory*，是一个由单参数 $p$ 控制的 gate-dependent circuit-level depolarizing model。[SI1]

它与 Helios 的差异不是简单的 Pauli composition 差异：

- native 2Q gate 是 `CZ`，不是 `SQRT_ZZ`/Helios CX lowering；
- 2Q/1Q/reset/measurement/idle 使用固定比例 $p,p/10,2p,5p$；
- measurement/reset 时间层会给其他 qubits 叠加较大的 waiting noise；
- 它不包含器件级 calibration-resolved X/Z bias；
- 它默认不解决具体超导芯片的 connectivity、routing、SWAP 或 leakage。

因此 SI1000 可以作为独立 reference campaign 或 shared DEM 基础设施的验证 profile，但不能放入 H-native/H-sym 的 2×2 cross-evaluation 并声称三者只相差 Pauli composition。

### 6.2 采用 modern SI1000 语义

原始论文定义的 SI1000 noisy gateset 为：[SI1,SI2]

| Physical location | Stim channel | Strength |
|---|---|---:|
| `CZ` 后 | `DEPOLARIZE2(p)` | $p$ |
| 任意 physical 1Q Clifford 后 | `DEPOLARIZE1(p/10)` | $p/10$ |
| Z-basis reset 后 | `X_ERROR(2p)` | $2p$ |
| Z-basis measurement 前 | `X_ERROR(5p)` | $5p$ |
| 当前 `TICK` 未参与物理操作的 qubit | `DEPOLARIZE1(p/10)` | $p/10$ |
| 存在 M/R 的 `TICK` 中，未被 M/R 的 qubit | 再叠加 `DEPOLARIZE1(2p)` | $2p$ |

ordinary idle 与 resonator idle 是两个独立 channel，必须叠加，而不是相加后直接写成 `DEPOLARIZE1(2.1p)`。后者没有正确处理两个 depolarizing channel 的复合概率，也不利于审计物理来源。

本项目采用后续文献明确的 **modern interpretation**：[SI3,SI4]

1. measurement 与 reset 是两个独立物理操作；
2. `M` 和 `R` 各自形成独立 noise moment；
3. 每个 M/R moment 上，其他 qubits 同时获得 `Idle(p/10)` 与 `ResonatorIdle(2p)`；
4. 不允许把 `MR` 当成一个只施加一次 idle/waiting noise 的合并操作。

在 artifacts 中必须保存：

```yaml
profile: si1000_modern_v1
definition: modern
source: arXiv:2108.10457v2
```

### 6.3 Stim 的职责边界

OmniQEC 固定使用的 Stim 1.15 提供 `DEPOLARIZE1`、`DEPOLARIZE2`、`X_ERROR`、noisy measurement 和 `Circuit.detector_error_model()`，但 Stim 本体没有 `stim.si1000(...)` 入口。常见的 `NoiseModel.si1000(p)` 来自 TQEC 等上层项目，可作为实现参考，不应被误记为 Stim 官方内置 profile。[SI5]

项目应在本地实现小型、严格、可测试的 `si1000_modern_v1` injector；Stim 负责：

- 模拟 noisy Clifford circuit；
- 将每个物理 fault 传播到 detectors/logical observables；
- 自动生成 candidate-specific DEM。

**禁止手写固定 SI1000 DEM。** 同一噪声参数在不同 $H_X,H_Z$、gate order、round count、detector definition 下会产生不同 DEM。

### 6.4 必须先构造带真实时间层的 noiseless CZ-native circuit

当前 `verification/circuit.py` 虽然生成了 edge-colored `schedule.rounds`，但随后通过 `schedule.cx_order()` 将 parallel layers 压平成一个列表。uniform noise 下这个问题不一定显现；SI1000 的 ordinary/resonator idle 完全依赖 `TICK`，因此必须重构为：

```python
for layer in schedule.rounds:
    append_native_two_qubit_layer(layer)
    circuit.append("TICK")
```

每个 physical moment 必须满足：

- 一个 qubit 在同一 moment 最多参与一个 physical operation；
- `DETECTOR`、`OBSERVABLE_INCLUDE`、`SHIFT_COORDS` 等 annotation 不算物理操作；
- M/R moment 不与 CZ/H moment 混合；
- 所有参与资源 accounting 的 data 与 ancilla 都显式加入 `system_qubits`，即使某个 qubit 在该 moment 尚未被初始化或即将被 reset。

### 6.5 抽象 CX 到 CZ-native syndrome block

SI1000 reference circuit 的物理 gate set 固定为：

```text
R, M, 1Q Clifford, CZ
```

使用等价关系：

$CX(c,t)=H(t)\,CZ(c,t)\,H(t)$。

不能为每条 edge 机械保留两个 H；同一个 target 的连续 CX 可以把中间的 $HH=I$ 消去。V1 固定采用 **syndrome-block-level merging**：在一个相同 orientation 的 CX block 开始前对所有 targets 执行一次 H，所有 edge-colored CZ layers 完成后再执行一次 H。

对当前 circuit orientation：

- Z-check：抽象 `CX(data, ancilla)`，target 集合为 ancillas；
- X-check：抽象 `CX(ancilla, data)`，target 集合为被 checks 触及的 data qubits；
- X-basis data preparation/measurement 必须编译到 Z-basis native operations，例如 `RX = R; H`、`MX = H; M`；
- 每个 H layer 和 CZ layer 之间放置 `TICK`；
- V1 不进行跨 syndrome-block、跨 measurement/reset boundary 的额外 Clifford cancellation，避免 candidate-dependent compiler ambiguity。

任何进一步优化必须版本化为新的 synthesis profile，并重新生成 circuit hash、DEM hash 与基准 LER。

### 6.6 初始化、每轮 measurement/reset 与终止边界

推荐的 reference memory protocol：

1. 初始时同一 moment reset 所有 data 与 ancillas；
2. 若为 X-memory，对 data 执行 native H preparation；
3. 执行第一轮 syndrome CZ block；
4. 单独 M moment 测量 ancillas；
5. 若仍有下一轮，单独 R moment reset ancillas；
6. 最后一轮后不再 reset 不会被使用的 ancillas；
7. 对 data 执行 basis-aware terminal measurement。

这样每轮恰好有一次 syndrome measurement，并由第一次全局 reset 或轮间 reset 提供下一轮 ancilla initialization；不能出现旧 circuit 中 `MR` 后下一轮又 `R` 所造成的重复 reset。

边界默认采用 `noisy_boundaries: true`，即初始 reset 和最终 data measurement 都使用 SI1000 channel。若为了复现使用 perfect time boundaries 的特定文献结果，必须作为独立 profile 记录，不能与 noisy-boundary LER 混合排名。

### 6.7 逐 moment 注噪算法

建议新增：

```text
verification/noise.py
```

配置对象：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SI1000NoiseConfig:
    p: float
    version: str = "modern"
    noisy_boundaries: bool = True

    def validate(self) -> None:
        if not 0 <= self.p <= 0.2:
            raise ValueError("SI1000 requires 0 <= 5p <= 1")
        if self.version != "modern":
            raise ValueError("Unsupported SI1000 interpretation")
```

injector 必须以 noiseless physical circuit 的 `TICK` 为边界。对每个 moment：

```python
def apply_si1000(circuit, *, cfg, system_qubits):
    out = stim.Circuit()

    for moment in split_physical_moments(circuit):
        physical_ops, annotations = separate_annotations(moment)

        # DETECTOR/OBSERVABLE/SHIFT_COORDS-only groups do not advance
        # physical time and therefore must not create an idle-noise layer.
        if not physical_ops:
            out += annotations
            continue

        touched = set()
        measured_or_reset = set()
        before = stim.Circuit()
        ideal = stim.Circuit()
        after = stim.Circuit()

        for op in physical_ops:
            qs = qubit_targets(op)
            assert touched.isdisjoint(qs)
            touched |= qs

            if op.name == "CZ":
                ideal.append(op)
                after.append("DEPOLARIZE2", op.targets_copy(), cfg.p)
            elif is_supported_1q_clifford(op):
                ideal.append(op)
                after.append("DEPOLARIZE1", sorted(qs), cfg.p / 10)
            elif op.name == "R":
                ideal.append(op)
                after.append("X_ERROR", sorted(qs), 2 * cfg.p)
                measured_or_reset |= qs
            elif op.name == "M":
                before.append("X_ERROR", sorted(qs), 5 * cfg.p)
                ideal.append(op)
                measured_or_reset |= qs
            else:
                raise ValueError(f"unsupported SI1000 physical op: {op}")

        idle = set(system_qubits) - touched
        if idle:
            after.append("DEPOLARIZE1", sorted(idle), cfg.p / 10)

        if measured_or_reset:
            waiting_for_mr = set(system_qubits) - measured_or_reset
            if waiting_for_mr:
                after.append("DEPOLARIZE1", sorted(waiting_for_mr), 2 * cfg.p)

        out += before
        out += ideal
        out += after
        out += annotations
        out.append("TICK")

    return out
```

上面 `waiting_for_mr` 必须是 `system_qubits - measured_or_reset`，不是普通 `idle` 的别名。这与原作者配套实现一致；即使未来允许其他 unitary 与 M/R 共时，未被 M/R 的 qubit 仍要获得额外 $2p$ channel。[SI2]

`split_physical_moments` 必须递归处理 `REPEAT` blocks，或者 V1 直接要求输入 circuit 已按有限 rounds 展开；不允许静默跳过 repeat body。annotation-only groups 不代表物理时间推进，不能凭空产生全系统 idle noise。

measurement 也可以写为 `M(5p)`，它在当前“ancilla 随后 reset / data terminal readout”的使用方式下与 `X_ERROR(5p); M` 具有相同 detector/observable statistics。为与原始论文定义逐项对齐，canonical artifact 使用显式的 `X_ERROR(5p); M`。

### 6.8 从 noisy circuit 自动生成 SI1000 DEM

对当前 BP+LSD、BP+OSD、RelayBP pipeline，canonical SI1000 DEM 使用：

```python
dem = noisy_circuit.detector_error_model(
    decompose_errors=False,
    flatten_loops=True,
    allow_gauge_detectors=False,
    approximate_disjoint_errors=False,
)
```

理由：

- SI1000 使用 `DEPOLARIZE1`、`DEPOLARIZE2` 和 `X_ERROR`；在物理使用的小概率范围内，Stim 可将 depolarizing channels 精确转换成独立 DEM mechanisms，不需要 `approximate_disjoint_errors`。[S1,SI6]
- 当前 BP-family decoders 可以让每个未分解 DEM mechanism 直接成为 $H$ 的一列；`decompose_errors=False` 保留一次物理 fault 造成多 detector 翻转的 hyperedge correlation。
- `flatten_loops=True` 避免当前 `dem_to_matrices()` 漏掉 `repeat` blocks 内的 errors。

如果未来加入 PyMatching/MWPM，应另外生成 decoder-specific graphlike DEM：

```python
matching_dem = noisy_circuit.detector_error_model(
    decompose_errors=True,
    flatten_loops=True,
    approximate_disjoint_errors=False,
)
```

禁止把 graphlike decomposition DEM 未经 separator/mod-2 处理直接交给当前 `dem_to_matrices()`。

### 6.9 当前固定 reference point

现有 uniform slow-loop 默认 `p=0.005` 不能直接继承。若解释为 SI1000，它对应：

- CZ：$0.5\%$；
- measurement：$2.5\%$；
- reset：$1.0\%$；
- resonator idle：$1.0\%$。

这可能使大量候选接近或超过有用工作区。因此当前 SI1000 reference campaign 不做多点 sweep，只固定论文常用 reference point：

```yaml
si1000_reference_p: 1.0e-3
```

$p=10^{-3}$ 是文献常用 reference point。当前结果不进行 SI1000 slope 拟合，也不把它解释为具体芯片 calibration。任何 uniform-noise protocol 下得到的 LER-slope offset、threshold 或 decoder calibration 都不能复用。

### 6.10 SI1000 正确性与验收测试

必须通过：

1. **论文 golden test**：复现原作者 `noise_test.py` 中的 `CZ`、1Q、M、R、idle 与 resonator-idle 小电路输出。[SI2]
2. **native lowering 等价**：用 Stim tableau/flow 验证 abstract CX circuit 与 noiseless CZ-native circuit 等价。
3. **moment invariant**：同一 `TICK` 内每个 physical qubit 最多参与一个物理操作，M/R 不与 unitary 混层。
4. **modern M/R test**：不存在 `MR`；M 与 R 两个 moments 都分别叠加 ordinary/resonator idle。
5. **zero-noise test**：$p=0$ 时 detector reference sample 与 abstract circuit 一致，DEM 不包含非零 mechanisms。
6. **channel-count audit**：保存 CZ、1Q、reset、measurement、ordinary-idle、resonator-idle 的 location counts。
7. **direct-vs-DEM sampling**：小码上比较 noisy circuit detector sampling 与 DEM sampling 的 detector/observable marginals，并要求在统计置信区间内一致。
8. **measurement representation test**：验证 `X_ERROR(5p); M` 与 `M(5p)` 在当前 protocol 下的 DEM 等价。
9. **fixed-seed reproducibility**：相同 code/schedule/config 生成相同 circuit hash、DEM hash 和 LER sampling stream。
10. **no-double-counting**：base circuit 不含旧 uniform noise；injector 遇到已有 noise instructions 时直接拒绝，不能静默清除或叠加。

### 6.11 SI1000 配置与 artifacts

```yaml
reference_noise:
  enabled: false
  profile: si1000_modern_v1
  role: independent_reference_campaign
  source_paper: arXiv:2108.10457v2
  interpretation: modern
  si1000_reference_p: 1.0e-3
  noisy_boundaries: true

reference_synthesis:
  native_gate_set: [R, M, H, CZ]
  cx_lowering: cz_block_merge_v1
  preserve_edge_coloring_ticks: true
  routing_model: off

reference_dem:
  decoder_family: bp
  decompose_errors: false
  flatten_loops: true
  allow_gauge_detectors: false
  approximate_disjoint_errors: false
```

每次 SI1000 run 至少保存：

- code hash 与完整 $H_X,H_Z$ characterization；
- noiseless native circuit hash；
- noisy circuit hash；
- synthesis profile/version；
- SI1000 config 与 $p$；
- 六类 physical noise location counts；
- DEM hash、mechanism/detector/observable counts；
- decoder config、shots、seed、LER 与 confidence interval。

SI1000 初版的准确 claim 是：

> SI1000 circuit-level noise-aware code discovery on OmniQEC's fixed abstract syndrome connectivity.

在没有二维 placement、nearest-neighbor routing 和 SWAP overhead 前，不得表述为 exact superconducting hardware implementation。

---

## 7. 初版为什么不建模 routing

Helios paper 报告的约 55 ms 是 98 qubits、每层随机重配对、fully dense 2Q gate 的 system-level benchmark 平均时间，不是任意 qLDPC syndrome layer 的固定时长。[H1]

在当前 Stim pipeline 中：

- `TICK` 和额外等待本身不产生 decoherence；
- “最多 4 个并行 2Q gates”若只导致 gate serialization，而不显式加入 idle/transport errors，则不会改变 DEM；
- routing failure 在 Helios all-to-all runtime 下通常不是静态 connectivity feasibility 问题；
- routing-dependent memory error 必须依赖一个可信的 candidate-specific transport schedule。

因此在没有 Helios compiler trace 或可验证 routing surrogate 前，把 55 ms 或 4-gate parallelism 直接转换为每层 decoherence 会引入比所研究 code effect 更大的建模任意性。

### 7.1 可选扩展层级

- **V1（主实验）**：native gate Pauli + 1Q + reset + measurement；无 routing。
- **V1.1（仅 sensitivity）**：每 round 加统一可扫的 `Z_ERROR(p_mem)`，明确不称为 exact Helios transport model。
- **V2（未来）**：由官方 compiler/emulator 或可信 QCCD compiler 输出 candidate-specific transport depth，再加入 transport dephasing、MCMR crosstalk 和 leakage。

Helios paper 的 transport-1QRB 结果可作为 V2 标定点：transport depth 1、2、4、8 的平均 infidelity约为 $6.0\times10^{-4}$、$1.28\times10^{-3}$、$3.4\times10^{-3}$、$8.4\times10^{-3}$。[H1] 但这些数据不能在没有 schedule 的情况下直接逐 edge-color 套用。

---

## 8. 推荐配置文件

```yaml
experiment:
  name: helios_code_discovery_v1
  physical_qubit_budget: 98
  families:
    - gb
    - tb
    - hgp
    - general_lp
    - btlp
  primary_metric: per_logical_per_round_ler

search:
  iterations: SAME_AS_EXPERIMENT_1
  programs_per_iteration: SAME_AS_EXPERIMENT_1
  seeds_per_program: SAME_AS_EXPERIMENT_1
  fast_proxy: kd2_over_n
  max_check_weight: 8

synthesis:
  protocol: one_ancilla_per_check_edge_coloring
  native_lowering: helios_full_clifford_v1
  lower_h_rx_mx: true
  ancilla_round_boundary: mr_then_init_no_duplicate_reset
  routing_model: off

noise:
  profile: helios_calibration_informed_pauli_v1
  helios_primary_scale: 1.0
  source_paper: arXiv:2511.05465v1
  emulator_profile: Helios-1E_2025-11-18
  p1: 2.5e-5
  p_init: 5.0e-4
  p_meas: 1.0e-6
  rzz_pauli_order:
    [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
  rzz_pauli:
    [4.5e-5, 4.5e-5, 19e-5, 5.8e-5, 0.06e-5,
     0.06e-5, 5.8e-5, 5.8e-5, 0.06e-5, 0.06e-5,
     5.8e-5, 19e-5, 4.5e-5, 4.5e-5, 5.9e-5]
  rzz_operand_orientation: control_first
  p2_scalar_applied: false
  leakage: off
  spontaneous_emission: reduced_into_pauli_surrogate
  transport_dephasing: off
  mcmr_crosstalk: off

control_campaign:
  profile: helios_symmetrized_v1
  preserve_total_rzz_fault_probability: true

dem:
  canonical_decoder: bp_family
  decompose_errors: false
  flatten_loops: true
  allow_gauge_detectors: false
  approximate_disjoint_errors: 1.0e-3

evaluation:
  decoder: SAME_AS_EXPERIMENT_1
  rounds: SAME_AS_EXPERIMENT_1
  cross_evaluate_top_k: 20
  search_sampling: approximate_dem
  final_sampling: exact_noisy_circuit
  report_dem_vs_direct: true
  orientation_sensitivity:
    - control_first
    - target_first
    - swap_average
```

### 8.1 N≤40 预实验固定配置

预实验的“40”约束的是 data-qubit code length proxy，而现有 `--n-budget` 统计 data + measurement ancilla，因此命令行同时固定：

```text
--n-budget 98 --proxy-n-max 40
```

两个独立 campaign 不做 sweep：

```yaml
si1000:
  profile: si1000_modern_v1
  si1000_reference_p: 1.0e-3

helios:
  profile: helios_calibration_informed_pauli_v1
  helios_primary_scale: 1.0
```

本阶段 CSV 只保留 metrics 列，例如 family、noise profile/axis/value、LER、shots、failures、CI、`d_proxy` 与 exact distance；不写 $A/B/L$、H-mask 或矩阵。完整结构与 provenance 留在 JSON artifact。

---

## 9. 实现顺序与验收标准

### Phase A：family constructors

1. 规范化 general HGP/LP 名称，并保留旧 `two_seed_*` aliases；
2. 实现共享 LP constructor 上的 fixed sector-H `btlp` metadata；
3. 用随机小实例验证 CSS ancestor，并对 BTLP 验证一般 symplectic commutation；
4. 验证 rank、$k$、$N_{\mathrm{phys}}$、weights、degrees 与 local-Clifford invariants；
5. 确认 program serialization、alias migration 与 deformation-aware hash 可复现。

### Phase B：shared native-circuit / DEM infrastructure

1. 生成完全 noiseless base circuit；
2. 保留 edge-colored `schedule.rounds`，为每个物理 layer 写入 `TICK`；
3. 拆除当前 uniform noise 与压平 `cx_order` 的耦合；
4. 按 noise/synthesis profile 生成 decoder-specific DEM；
5. 保存 noiseless/noisy circuit hash、DEM hash 与 noise-location counts。

### Phase B-SI：modern SI1000 reference evaluator

1. 实现 `CZ` syndrome-block lowering；
2. 实现逐 moment gate/reset/measurement/idle/resonator-idle injector；
3. 强制拆分 M/R，禁止把 `MR` 当成单一 noise moment；
4. 使用未分解、无 disjoint approximation 的 canonical BP DEM；
5. 完成第 6.10 节全部 SI1000 tests。

### Phase B-H：Helios noise injector

1. 实现 `H/RX/MX/CX/CZ/XCX` 的完整 Helios-native Clifford lowering；
2. 在每个 physical `Rxy` 后注入简化 1Q noise，在 virtual $R_Z$ 后不注噪；
3. 实现 H-native 和 H-sym 15-outcome channels，禁止额外叠加 scalar `p2`；
4. 重写 ancilla round boundary，确保相邻 rounds 之间恰好一次 initialization fault；
5. 实现 canonical undecomposed BP DEM；
6. 实现 exact noisy-circuit detector sampling，并保存 DEM-vs-direct 对照；
7. 实现 `control_first`、`target_first` 与 `swap_average` orientation modes；
8. 确认没有残留原 uniform depolarizing noise；
9. 确认 SI1000 与 Helios 使用独立、完整版本化的 noise/synthesis configs。

### Phase C：正确性测试

必须通过：

- shared schedule 中每个 edge-color layer 都有明确 `TICK`，同一 qubit 不会在同一物理 moment 被重复操作；
- 所有 noise profiles 都从同一个无旧 noise 的 base representation 开始，避免 double counting；
- SI1000 通过第 6.10 节的 golden、modern M/R、direct-vs-DEM 与 reproducibility tests；
- noiseless native CX sequence 与 `CX` 的 tableau/flow 等价；
- noiseless native CZ/XCX sequence 分别与 `CZ`/`XCX` 的 tableau 等价；
- noiseless native H sequence 与 `H` 的 tableau/flow 等价；
- 所有抽象 `H/RX/MX/CX/CZ/XCX` 都被 lowering，最终 Helios circuit 不残留未定义的抽象 basis operation；
- 每个 physical `Rxy` location 恰好一次 1Q noise，每个 virtual $R_Z$ location 为零次；
- 首次 ancilla preparation 和每个 round boundary 的 initialization-noise count 与预期一致，不存在 `MR` 后紧接冗余 `R`；
- H-native 15 项之和为 $8.534\times10^{-4}$；
- H-sym 15 项之和完全相同；
- native 2Q channel 与 scalar `p2` 不会同时出现；
- `control_first`、`target_first` 与 `swap_average` 的 Pauli-label transform 通过 golden-vector test；
- noise scale 为 0 时，native-lowered circuit 的 detector reference sample 与原 circuit 一致；
- 所有候选 DEM 可生成；
- canonical BP DEM 为 `decompose_errors=False`，不含 decoder parser 无法解释的 component separator；
- DEM sampling 与 direct circuit detector sampling 的 detector/observable marginals 和最终 LER 在统计置信区间内一致；
- 固定 random seeds 后结果可复现；
- H-native/H-sym 只在 RZZ Pauli composition 上不同；
- BTLP X/Z memory 在零噪声下 detector/observable 均不翻转，且 SI1000/Helios noisy DEM 均可生成；
- `general_lp` 与同 $A/B/L$ 的 `btlp` ancestor 矩阵相同、compiled hash 不同，BTLP CSV 不伪报 `d_x_proxy/d_z_proxy`；
- 五个默认 family 均通过 fast loop、schedule、SI1000/Helios DEM、slow verifier、Sinter 与 CSV smoke test；
- uniform legacy circuit/DEM golden hash 和现有 CSS SI/Helios 行为保持不变。

### Phase D：campaign 与结果

1. 先用小预算/少迭代 smoke test；
2. 再固定 98-qubit budget 跑 H-native 与 H-sym；
3. 保存每次 iteration 的 program、code properties、circuit hash、noise config、DEM hash 和 LER；
4. 对 top-$K$ 做 native/sym cross-evaluation，并使用 exact noisy-circuit sampling 确认；
5. 对 top-$K$ 做 `control_first`/`target_first` orientation sensitivity；若不稳定，再加入 `swap_average`；
6. 当前不做 error-scale sweep；最终 codes 继续固定 `scale=1` 并使用 direct-circuit LER。

---

## 10. 建议输出图表

### 主图 A：family-aware discovery result

- 横轴：campaign（H-sym / H-native）或 iteration；
- 纵轴：$-\log_{10}(\mathrm{LER})$；
- 颜色：family；
- 标注：$[[n,k,d]]$、$d_X/d_Z$、$w_X/w_Z$。

### 主图 B：cross-evaluation

四组 box/point distributions：

- native-searched → native-evaluated；
- native-searched → sym-evaluated；
- sym-searched → native-evaluated；
- sym-searched → sym-evaluated。

### 辅助表

| Code | Family | $N_{\mathrm{phys}}$ | $[[n,k,d]]$ | $d_X,d_Z$ | $w_X,w_Z$ | LER native | LER sym | Orientation sensitivity |
|---|---|---:|---|---|---|---:|---:|---:|

### 结果 punchline 的理想形式

> Under an equal total two-qubit fault probability, the structured Helios native Pauli channel changes the family and construction parameters favored by circuit-level feedback. OmniQEC discovers asymmetric product-code instances that outperform codes found under the symmetrized control when both are evaluated under the Helios-calibration-informed model.

该句是目标假设，不应在结果出来前预写成既定结论。

---

## 11. 已知限制与需要记录的设计决定

1. **2Q Pauli channel**：来自 2QCB，存在 symmetry assumptions 和不可学习自由度；不是完整 process tomography。
2. **Leakage**：A4 Pauli table 不包含 persistent leakage；初版关闭，不应声称复现完整 2Q infidelity。
3. **1Q channel**：完整 $X/Y/Z$ composition 未公开；`DEPOLARIZE1(p1)` 是工程近似。
4. **SPAM split**：`p_init` 与 `p_meas` 来自 emulator profile，不是 paper 的独立 preparation/readout benchmark。
5. **1Q/SPAM emulator gap**：官方 emulator 还包含 asymmetric depolarization、spontaneous emission 和 leakage；V1 只使用 Pauli surrogate。[H4]
6. **RZZ operand orientation**：routing-independent circuit 不知道 abstract control/target 到 physical ion order 的真实映射；必须做 swapped-orientation sensitivity。
7. **MCMR crosstalk**：与 batch、protected measurement 和 spectator location 有关；初版关闭。
8. **Transport memory**：依赖 candidate-specific schedule；初版关闭。
9. **DEM approximation**：`PAULI_CHANNEL_2` 转 DEM 使用 small-probability independent approximation；最终测试数据改由 exact noisy-circuit sampler 产生，但 decoder prior 仍是 independent approximation。
10. **DEM decomposition**：canonical BP path 使用 `decompose_errors=False`；graphlike DEM 只能作为独立 decoder-specific artifact，不能交给当前忽略 separator 的 parser。
11. **Model versioning**：Quantinuum 明确说明 emulator parameters 会更新；每次实验必须保存 profile date、access date 与完整参数快照。[H4]
12. **Claim boundary**：结果证明 noise-aware adaptation，不证明最终 code 已在 Helios 上完成 exact compilation 或 hardware execution。
13. **SI1000 model scope**：SI1000 是 gate-dependent superconducting-inspired model，不是具体超导芯片的 calibration model，也不提供 X/Z-biased device channel。
14. **SI1000 routing**：当前 abstract data–ancilla connectivity 未加入二维 placement、nearest-neighbor routing、SWAP 或 leakage；SI1000 结果不能称为 exact superconducting hardware implementation。
15. **Compiler dependence**：SI1000 DEM 依赖 CZ lowering、H cancellation、`TICK` 与 boundary policy；这些必须版本化，任何变化都要求重新生成 circuit/DEM hash 并重新标定 LER protocol。
16. **BTLP claim boundary**：BTLP 原论文的数值结果采用 perfect syndrome readout，并明确把 circuit-level biased-noise simulation 留作后续问题。[C4] OmniQEC v0.5 的 mixed-syndrome circuit、SI1000/Helios lowering 与 boundary policy 是新的、版本化的 V1 compiler convention，不应描述成“论文已验证的电路实现”。
17. **BTLP V1 范围**：只支持论文可解释的 fixed `sector_h` local Clifford image，不支持 arbitrary non-CSS code、任意 H-mask 或 pure-X/Z effective-distance optimizer；最终优劣由相同 slow-loop LER evaluator 决定。

---

## 12. 参考资料

- **[O1]** *OmniQEC: discovering practical quantum error-correcting codes by an AI scientist*, 当前上传稿正文，重点为 system architecture、dual-loop search、circuit synthesis 与 circuit evaluation 部分。
- **[H1]** A. Ransford et al., *Helios: A 98-qubit trapped-ion quantum computer*, arXiv:2511.05465v1. https://arxiv.org/abs/2511.05465
- **[H2]** Quantinuum Systems, *Helios Operation*. https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/helios.html
- **[H3]** Quantinuum Systems, *Quantinuum Helios — Default Noise Model Parameters*. https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/helios_emulators.html
- **[H4]** Quantinuum Systems, *Emulator Noise Model*. https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html
- **[S1]** Quantumlib, *Stim documentation and supported gates*. https://github.com/quantumlib/Stim/blob/main/doc/gates.md
- **[SI1]** C. Gidney, M. Newman, A. Fowler, and M. Broughton, *A Fault-Tolerant Honeycomb Memory*, Quantum 5, 605 (2021), arXiv:2108.10457v2. https://arxiv.org/abs/2108.10457
- **[SI2]** 同上论文的作者配套 SI1000 实现与 tests：`noise.py` / `noise_test.py`. https://arxiv.org/src/2108.10457v2/anc/repo/src/noise.py 和 https://arxiv.org/src/2108.10457v2/anc/repo/src/noise_test.py
- **[SI3]** C. Gidney, M. Newman, P. Brooks, and C. Jones, *Yoked surface codes*, arXiv:2312.04522；其 SI1000 规范用于澄清 modern interpretation. https://arxiv.org/abs/2312.04522
- **[SI4]** A. W. Senior et al., *A scalable and real-time neural decoder for topological quantum codes*, arXiv:2512.07737v2；明确说明 measurement/reset 分离及 ordinary/resonator idle 的 modern stacking. https://arxiv.org/abs/2512.07737
- **[SI5]** TQEC, `NoiseModel.si1000` documentation；作为 Stim 上层 noise injector 的实现参考。 https://tqec.github.io/tqec/_autosummary/tqec.utils.NoiseModel.html
- **[SI6]** Quantumlib, *Stim error analysis / detector error model documentation*. https://github.com/quantumlib/Stim/blob/main/doc/usage_command_line.md 和 https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md
- **[C1]** J.-P. Tillich and G. Zémor, *Quantum LDPC codes with positive rate and minimum distance proportional to the square root of the blocklength*, arXiv:0903.0566.
- **[C2]** P. Panteleev and G. Kalachev, *Quantum LDPC Codes with Almost Linear Minimum Distance*, arXiv:2012.04068.
- **[C3]** *Multivariate Bicycle Codes*, arXiv:2406.19151. https://arxiv.org/abs/2406.19151
- **[C4]** J. Roffe et al., *Bias-tailored quantum LDPC codes*, arXiv:2202.01702. https://arxiv.org/abs/2202.01702
- **[C5]** 作者官方 BTLP 实现，`quantumgizmos/bias_tailored_qldpc`. https://github.com/quantumgizmos/bias_tailored_qldpc

---

## 13. 最终实现摘要

```text
LLM construction program
    -> gb / paper tb / hgp / general_lp / btlp
    -> CompiledCode(CSS ancestor + optional sector-H)
    -> CSS or symplectic structural validation
    -> local-Clifford-invariant fast proxy and 98-qubit filtering
    -> CSS or mixed-stabilizer syndrome schedule with explicit parallel TICKs
    -> choose a versioned physical synthesis/noise profile
       -> SI1000 reference:
          CZ-native syndrome blocks with deformation-aware H frames
          + modern gate/reset/measurement/idle/resonator-idle channels
       -> Helios main experiment:
          lower H/RX/MX/CX/CZ/XCX to Helios native Clifford sequence
          + inject H-native or H-sym Pauli channels
          + fix one initialization fault per round boundary
          + version RZZ operand orientation
    -> canonical undecomposed Stim DEM -> decoder H/L/priors
    -> approximate DEM sampling for search
    -> exact noisy-circuit detector sampling for top-K/final LER
    -> direct Sinter LER and metrics-only CSV
    -> circuit-level evidence returned to OmniQEC
```

**核心开发原则：**搜索变量仍然是 code family 与 family parameters；BTLP 只增加一个不可任意扩张的 `sector_h` family choice。SI1000/Helios 只通过固定、完整版本化、可复现的 physical synthesis 与 noise evaluator 进入 slow loop。两者不能共享含义不清的标量 `p`，也不能共用未记录 profile 的 DEM。routing、transport 和 calibration optimization 不进入 LLM search space。

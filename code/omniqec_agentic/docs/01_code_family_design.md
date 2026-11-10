# OmniQEC Fig.2 重构设计定稿 01：Code Family

- 状态：**定稿**（2026-08-15；v1.1 修订同日）
- 依据：《OmniQEC New Fig.2 Experiment：Code Family 与 Slow-Loop 重构规划》PDF + 本次讨论决策 + 文献调研
- 修订记录：v1.1 —— LP shape 池开放 `2×2`（用户决策，覆盖 PDF"暂不开放 2×2"的初版建议）；新增 LP seed-shape 钉定开关（§3.5）
- 范围：`construction/`、`deformation/` 及其对 registry/prompt/校验的接口。SEC、decoder、slow loop 属于后续设计稿（02+）。
- 替代关系：**全量替代**现有 9 个构造族（`gb/tb/hgp/general_lp/btlp/lp/gb3/2bga/glp`）及全部兼容别名。

---

## 1. 定位与原则

1. 不追求数学意义上的最大覆盖空间，而是四个具有明确 inductive bias 的 **structured search grammars**。对外表述统一用 "construction spaces / search grammars"，不称互斥的 code families（General BB 与 Abelian LP 数学上存在包含关系，作为强结构先验单独搜索是合理的）。
2. 四个 grammar 共享同一 slow-loop 评估预算、`N_phys` 网格、`w_max ≤ 9`、噪声模型、SEC、decoder——实验结论是 **grammar/search-bias 比较**，不是算力比较。
3. `lp_abelian` 与 `lp_nonabelian` 构成受控实验（Mitten 论文动机：1×2 LP 结构下阿贝尔群存在距离瓶颈，非阿贝尔群突破它），因此二者必须**共用同一套 seed-shape 和 weight grammar 的代码路径**，仅群来源不同——由构造保证，不靠纪律保证。

## 2. 四个 Family 总览

| 注册名 | Seed 结构 | 群 | 主要 inductive bias |
|---|---|---|---|
| `bb` | 一对 1×1 群代数元素 A/B（shape 固定） | `C_ℓ × C_m` | 强 bicycle/代数结构 |
| `hgp` | 两个独立 classical parity-check 矩阵 H1/H2 | 无 | classical-seed product |
| `lp_abelian` | 一对 base matrices，shape 各自独立选自 shape 池 | 全体有限阿贝尔群 | seed + 交换 lift |
| `lp_nonabelian` | 与 `lp_abelian` 完全同一 grammar | 非阿贝尔群目录 | seed + 非交换 lift |

`construct(rng)` 候选程序协议不变：返回 Python dict、随机性来自传入的 `numpy.random.Generator`、返回普通 int。

## 3. 各 Family 规范

### 3.1 `bb` — General BB

在 two-block CSS skeleton `HX = [A|B]`，`HZ = [Bᵀ|Aᵀ]` 上，允许 A、B 是 `Z_ℓ × Z_m` 上任意 sparse bivariate polynomials。复用现有 `gb.py` 的多项式 builder，注册名统一改为 `bb`。

```python
{"family": "bb", "ell": 6, "m": 6,
 "A": [[3, 0], [0, 1], [0, 2]],   # 每项 [a, b] 表示 x^a y^b
 "B": [[0, 3], [1, 0], [2, 0]]}
```

约束与规范化：

- weight grammar：`4 ≤ len(A) + len(B) ≤ 9`（即 `4 ≤ w_A + w_B ≤ 9`；check weight = `w_A + w_B`）；
- A、B 内部不允许重复项（F2 上重复即消项，视为退化候选直接拒绝）；
- 指数按 `mod ℓ / mod m` 规范化；
- **`ℓ ≤ m` 规范化**：validator 遇到 `ℓ > m` 时自动交换两轴（`ell↔m`，每个 `[a,b]↔[b,a]`），并在 metadata 记录发生过规范化；
- validity：`k > 0`（全局检查）、无 zero/duplicate checks、CSS commute。

规模：`N_data = 2ℓm`，`N_checks = 2ℓm`，`N_total = 4ℓm`。

### 3.2 `hgp` — Hypergraph Product

经典 Tillich–Zémor 构造。两个 seed 完全独立，由 LLM 分别设计：

```text
HX = [H1 ⊗ I | I ⊗ H2ᵀ],   HZ = [I ⊗ H2 | H1ᵀ ⊗ I]
```

```python
{"family": "hgp",
 "seeds": {"H1": [[1,1,0,1], [0,1,1,0], ...],
           "H2": [[...], ...]}}
```

约束：

- 每个 seed 独立过 `grammars/classical_seed.py`：0/1 矩阵；**行度与列度均在 `[2, 4]`**（初始范围，配置项）；拒绝零行/零列与完全重复的行；
- **不做人为 shape 限制**——seed 尺寸由 `N_phys` 预算公式自动约束；
- 允许 `H1 == H2`；允许 irregular seed、sparse random / QC-inspired / LLM 手写结构；
- **不要求 full-row-rank**；直接构造后算 k，只要求全局 `k > 0`；
- 编译后强制 `w_X, w_Z ≤ 9`（度数范围 2–4 时自动满足 ≤ 8，双保险）。

规模（H1: `m1×n1`，H2: `m2×n2`）：`N_data = n1·n2 + m1·m2`，`N_checks = m1·n2 + n1·m2`，`N_total = (n1+m1)(n2+m2)`。

### 3.3 `lp_abelian` — Small-seed Abelian LP

保留 group lift，但 protograph seed 限制得非常小；搜索发生在 small seed structure + Abelian group + sparse group-element supports 三个维度。LP 构造语义沿用现有 glp builder（Lin–Pryadko `LP[a,b]` 约定，左/右正则表示；1×1 退化为 2BGA 区域）——**限制现有 GLP grammar，不重写 LP engine**。

```python
{"family": "lp_abelian",
 "group": {"factors": [4, 3]},            # G = C4 × C3
 "A": [[ [[1,0],[0,2]], [[2,1]] ]],       # 1×2；cell = 指数元组列表
 "B": [[ [[0,0],[1,1],[3,2]] ]]}          # 1×1；[0,0] 即单位元
```

约束：

- **shape 池**：`allowed_shapes = [[1,1],[1,2],[2,1],[2,2]]`（配置项，最大 `2×2`）。**A、B 的 shape 各自独立选择**（共 16 种组合），由 LLM 根据 shape 池上限和预算公式自主判断；shape 的运行级钉定见 §3.5；
- `group.factors`：任意 `q_i ≥ 2` 的列表，`|G| = ∏ q_i` 受预算约束；**不只搜 cyclic group**。构造按声明的 factors 逐字执行；metadata 中记录不变因子规范形作为群的同构不变标识；
- cell = G 中元素的稀疏子集，以指数元组列表表示；**cell 不允许为空**（空 cell 是变相降 shape，破坏 shape 统计的诚实性），cell 内元组不允许重复；
- **supports 生成整个 G**：A、B 中出现的全部元素生成的子群必须等于 G（避免实际上只用了更小 subgroup；也提前给出比全局连通性检查更可解释的失败信息）；
- 编译后强制 max check weight ≤ 9（grammar 层另设宽松的 per-cell weight 上限 `[1, 8]` 作为 prompt 约束；8 是 1×1 shape 下合法码可用的最大 cell weight，如 Lin–Pryadko 2BGA 的 w2+w6 组合，真正仲裁是编译后 w ≤ 9 硬检查）。

规模（A: `mA×nA`，B: `mB×nB`）：`N_data = (nA·nB + mA·mB)·|G|`，`N_checks = (mA·nB + nA·mB)·|G|`，`N_total = (nA+mA)(nB+mB)·|G|`。

### 3.4 `lp_nonabelian` — Small-seed non-Abelian LP

与 `lp_abelian` **完全同一 grammar 与 builder 代码路径**，唯一差异是群来源：

```python
{"family": "lp_nonabelian",
 "group": "C5xS3",                        # 目录名（见 §4.2）
 "A": [[ [[0,0,0],[2,1,0],[4,0,1]], [[1,1,1]] ]],
 "B": [[ [[0,0,0],[3,0,1],[1,2,0]] ]]}
```

附加校验：

- supports 生成整个 G（同 Family 3；G 非阿贝尔时该条自动蕴含 `⟨supp(A), supp(B)⟩` 真非阿贝尔）；
- 显式断言生成子群非阿贝尔（与生成性检查冗余，但保留为独立断言，防止后续放松生成性约束时静默失去该保证）。

`group` 字段语法与 Family 3 不同（目录名 vs factors）是**有意的**：群来源正是受控变量，PDF 要求一致的是 seed-shape 与 weight grammar，这两者由共用代码路径保证。

### 3.5 LP seed-shape 钉定开关（`lp_abelian` / `lp_nonabelian` 共用）

运行级配置，对 A、B **各自独立**给出可选的 shape 钉定：

```yaml
lp_shape_pin:
  A: [1, 2]     # 钉定 A 的 shape 为 1×2；null 表示不钉定
  B: null       # B 的 shape 由 LLM 在 shape 池内自选
```

语义：

- 钉定只约束 **shape**，被钉定矩阵的 cell 内容仍由 LLM 设计；
- 两个字段独立取值，自然覆盖三种搜索模式：
  - **two-seed（默认）**：`A: null, B: null`——A、B 的 shape 都由 LLM 自选（对应 HGP 的双 seed 自由度）；
  - **one-seed**：钉定其一（如 `A: [1,2]`），另一个自由——把搜索半约束在目标结构邻域，同时保留 LLM 对另一半的探索空间；
  - **fully pinned**：两个都钉定（如 `A: [1,2], B: [1,2]`）——精确锁定结构网格，即 Mitten 码的 1×2/1×2 形态；
- prompt 渲染：被钉定的矩阵在 genotype 说明中直接写死其 shape 为硬性要求，自由矩阵展示 shape 池与预算公式；
- validator 强制：spec 中被钉定矩阵的 shape 与配置不符时直接拒绝（在 grammar 层报错，反馈信息明确指出钉定要求）；
- 钉定配置属于**测量协议的一部分**，写入 run config 与产物 metadata，不同钉定设置的 run 结果不可直接混合比较。

## 4. 群目录（`construction/groups/`）

### 4.1 `abelian.py`

- 按预算枚举/校验有限阿贝尔群 `C_q1 × C_q2 × ...`；
- 提供不变因子规范化（如 factors `[2, 6]` 与 `[2, 2, 3]` 同构，规范形 `d1 | d2 | ...`），仅用于 metadata 标识与报告，不改写构造用的声明 factors；
- 元素 = 指数元组 `[e1, ..., ek]`，`e_i ∈ [0, q_i)`，乘法逐坐标模加。

### 4.2 `catalog.py` — 非阿贝尔目录

**目录组成（v1 定稿）**：

| 类别 | 生成规则 | 命名 |
|---|---|---|
| 二面体 | `D_n`，`n ≥ 3`（`D3 = S3`） | `D3`, `D4`, ... |
| 交错/对称 | `A4`、`S4`（**A5 移除**，理由见下） | `A4`, `S4` |
| Metacyclic | `C_m ⋊_k C_n`：`gcd(k,m)=1`，`k^n ≡ 1 (mod m)`，`k ≢ 1`，`m ≥ 3` | `C5:C20(3)`（括号内为 twist k） |
| 直积 | `C_m × H`，`H ∈ {D_n, A4, S4}`，`m ≥ 2` | `C5xS3`, `C4xD10`, `C13xA4`, ... |

设计要点：

- **目录按运行预算参数化生成**，不是硬编码列表：给定 `n_budget` 与 shape 池，可用的最大 `|G|` 由 `N_total = (nA+mA)(nB+mB)·|G| ≤ n_budget` 反推。288 预算下按 shape 组合：双 1×1 → `4|G|`（`|G| ≤ 72`）；1×1 配 1×2/2×1 → `6|G|`（`|G| ≤ 48`）；双 1×2（Mitten 形态）→ `9|G|`（`|G| ≤ 32`，C5×S3 的 30 恰好可达）；双 2×2 → `16|G|`（`|G| ≤ 18`）。预算增大时目录自动扩展（Mitten 更大的码如 [[500,100,16]] C5⋊C20 需要更大预算网格时可达）；
- **同构去重采用规则式**（保守策略）：已知同构关系显式处理（`C2 × D_n ≅ D_2n`（n 奇）；metacyclic 的 `k ~ k^j`（j 与 n 互素）等价类取最小代表；`D_n` 与 metacyclic `C_n ⋊_{-1} C_2` 的重叠归入 `D_n`），不做通用同构判定。指纹（阶、元素阶直方图、中心阶、导群阶、阿贝尔化不变量）仅作为测试断言辅助，指纹冲突时宁可保留重复也不错误合并；
- **GAP SmallGroup ID 作为 metadata**：每个目录项附 `(order, index)`（文献可查者填充，如 C5×S3 = (30,1)、C4×D10 = (40,5)、C5⋊C20 = (100,9)；未知者置 `null` 后续补），用于与 Lin–Pryadko / Mitten 已发表码表对读；
- **A5 移除**：它是候选群中唯一的非可解群（有限群 polycyclic ⟺ 可解），无指数正规形，单独维护置换编码会破坏 Family 3/4 的 spec 格式统一；Mitten 论文的全部群都是可解群，去掉 A5 不影响搜出 Mitten 码。若未来需要，可作为带独立编码的特例加回。

### 4.3 元素编码 — pc 正规形指数元组

**约定**：每个目录群声明一个有序生成元序列 `(g1, ..., gk)` 及各自相对阶 `(r1, ..., rk)`，使每个元素有唯一正规形 `g1^e1 · g2^e2 · ... · gk^ek`，`e_i ∈ [0, r_i)`；spec 中元素 = 指数元组 `[e1, ..., ek]`。这是 GAP 对可解群（PcGroup）的标准做法，阿贝尔情形是其交换特例——Family 3/4 的元素语法严格同构。

各类群的生成元约定：

| 群 | 生成元序列 | 元组含义 | 定义关系 |
|---|---|---|---|
| `C_q1 × ... × C_qk` | 各因子生成元 | `[e1..ek]` | 全交换 |
| `D_n` | `(r, s)`，阶 `(n, 2)` | `[i, j]` = `r^i s^j` | `s⁻¹ r s = r⁻¹` |
| `C_m ⋊_k C_n` | `(r, s)`，阶 `(m, n)` | `[i, j]` = `r^i s^j` | `s⁻¹ r s = r^k` |
| `A4` | `(a, b, c)`，阶 `(3, 2, 2)` | `[i, j, l]` = `a^i b^j c^l` | `V4 = ⟨b,c⟩` 正规，a 以共轭轮换 b,c,bc |
| `S4` | `(t, a, b, c)`，阶 `(2, 3, 2, 2)` | `[i, j, l, p]` | 沿合成列 `S4 ⊳ A4 ⊳ V4 ⊳ C2` |
| `C_m × H` | `C_m` 生成元前置 ++ H 的序列 | `[e] ++ tuple_H` | 直积 |

实现说明：元组是**对外接口**（spec、prompt、日志、报告）；builder 内部把每个群实体化为元素列表 + Cayley 表（由声明的关系一次性构建），正则表示矩阵从 Cayley 表生成。与文献机器格式（GAP `Elements(G)` 默认序整数索引，Lin–Pryadko 数据仓库与 Mitten Table XIII 所用）之间提供确定性转换工具，用于外部码表对读与回归测试。

## 5. Seed Grammar 模块（`construction/grammars/`）

- **`classical_seed.py`**：单个 classical parity-check 矩阵的语法校验（0/1、行列度范围、零行零列、重复行）。`hgp` 对 H1/H2 各调用一次。度数范围 `[2,4]` 为配置项。
- **`ga_seed.py`**：`F_2[G]` base-matrix 语法校验（shape ∈ `allowed_shapes`、shape 钉定符合 `lp_shape_pin`（§3.5）、cell 非空且元组不重复、指数范围合法、per-cell weight、supports 生成 G）。`bb` 以 `allowed_shapes=[[1,1]]` + `G = C_ℓ×C_m` 的退化配置复用同一模块；两个 LP family 以完整 shape 池 + 钉定配置实例化。
- shape 池、度数范围、weight 上限全部是**配置项**，registry 把它们连同预算公式一起渲染进 prompt——LLM 看到"在这些 shape 里根据预算自己选"，未来开放 2×2 只改配置不改代码。

## 6. 校验三层

```text
grammar 层（grammars/*）      seed 本身：shape、稀疏度、weight、生成性
      ↓
family 层（各 family 插件）    预算公式、群合法性、规范化（ℓ≤m 等）
      ↓
全局结构检查（verification/structural.py，保持现状 + 一处新增）
                              CSS commute、k>0、Tanner 连通、
                              N_data + rows(HX) + rows(HZ) ≤ N_budget、
                              【新增】max check weight ≤ w_max（默认 9，配置项）
```

注：现行 structural.py 刻意不设 check-weight 阈值；本次按 PDF 统一口径新增 `w_max ≤ 9` 硬检查（作为跨 family 公平性的一部分），默认值可由 config 覆盖。

## 7. Deformation 模块（`deformation/`）

- 独立模块，**运行级 config 开关**：`deformation.enabled`（默认 `false`）+ `deformation.mode`（v1 仅 `sector_h`）。**不进 genotype、不进 prompt**，LLM 不感知——保持四 grammar 的 CSS 比较不被额外搜索维度污染；开关留给实验 ablation。
- 四个 family 的 builder 都在 CompiledCode 中导出**双 sector 边界元数据**（bb：两个 `ℓm` block；hgp：`n1·n2 | m1·m2`；LP：`(nA·nB)|G| | (mA·mB)|G|`）。deformation 模块只消费元数据，完全 family 无关。
- `sector_h`：保持 CSS ancestor 的 `HX/HZ`，在第二 sector 交换物理稳定子 X/Z 轴（H-mask）。code hash 已纳入 H-mask，无需改动。
- 旧 btlp 的插件内建 H-mask 路径随旧 family 删除一并消失；原"CLI 运行级 mode 覆盖插件内建 mask"的矛盾不再存在。

## 8. 目标目录结构

```text
construction/
├── registry.py            # 4 个 family 注册：spec validator、builder、
│                          #   genotype 格式、语法示例、预算公式、prompt 约束
├── grammars/
│   ├── classical_seed.py
│   └── ga_seed.py
├── groups/
│   ├── abelian.py
│   └── catalog.py         # 非阿贝尔目录（预算参数化生成 + GAP ID metadata）
├── bb.py                  # 薄封装原 gb.py builder
├── hgp.py
├── lp.py                  # Abelian/non-Abelian 共用 builder，群来源为参数
└── build.py               # spec → CompiledCode（HX/HZ + sector 元数据）

deformation/
├── api.py                 # apply(compiled, config) -> CompiledCode
└── sector_h.py
```

## 9. 删除清单与迁移后果

删除：`gb.py`（内容迁入 `bb.py`）、`tb.py`、`two_seed_hgp.py`（重写为 `hgp.py`）、`two_seed_lp.py`、`gb3.py`、`twobga.py`（群目录迁出至 `groups/catalog.py` 并重组）、`lifted_product.py`、`glp.py`（builder 迁入 `lp.py`）、registry 中全部旧 family 与兼容别名。

连带清理（随后续 workflow 重构执行）：`config.py` 默认 family 集、`examples/` 内置种子、tests、hook surrogate 的 gb/lp 专用特征、`check-ver1-baseline`、CLI `--families` 默认值与文档。

已接受的后果：历史 run 的 spec 无法重新编译回放；对读历史结果只能依赖已存档的矩阵与指标。

## 10. 文献对读

- **Mitten 论文**：arXiv:2607.28795 "High-rate qLDPC processors"（Caltech + Oratomic, 2026-07-30）。Mitten codes = 1×2 LP over 非阿贝尔群，check weight 9、rate 20%；证明同 shape 阿贝尔群距离被 classical base matrix check weight 封顶。其群目录（C5×S3、C4×D10、C10×S3、C5×S4、C11×S3、C7×S4、C13×A4；C5⋊C20、C9⋊C12、C7⋊C18、C13⋊C15）全部可解、无 A5——本设计目录组成即为覆盖该疆域。旗舰 [[150,30,10]]（C5×S3，N_total=270）落在 288 预算网格内。构造数据在 Appendix K Table XIII（GAP ID + Elements(G) 索引编码）。
- **Lin–Pryadko 2BGA**：arXiv:2306.16400 / PRA 109, 022407 (2024)。群标识 = GAP SmallGroup ID 的事实标准来源；机器格式（github.com/QEC-pages/2BGA-codes）= 元素索引列表；论文表格用生成元正规形字（二面体 `s^ε r^k`）。
- **编码惯例结论**：学界群身份用 GAP ID、机器格式用元素索引、人类可读用呈现相关正规形（各家不统一）；可解群的指数向量正规形是 GAP/Magma 的标准内部编码。本设计取"指数元组为对外接口 + GAP ID 为 metadata + 提供索引转换工具"的组合。

## 11. 非阻塞待办

- 目录项 GAP ID 的批量回填（可用文献表格核对 + 后续有 GAP 环境时脚本补全）；
- metacyclic `k ~ k^j` 等价类归约与直积重叠去重的单元测试（以指纹断言辅助）；
- 与 Mitten Table XIII 的对读回归测试：把 [[150,30,10]] C5×S3 的索引编码转换为本设计元组编码，验证 builder 复现其 `n/k`（需要 GAP `Elements(G)` 排序的确定性转换工具）。

## 12. 实现验证记录（2026-08-15，实现即本文档定稿的落地）

- 实现落地说明：`bb` 保留独立多项式校验（与 `ga_seed` 语义一致，未强行复用退化配置）；非阿贝尔目录采用"名称语法 + 按需构造 + lru 缓存"实现预算参数化（非硬编码列表），metacyclic 非规范 twist 自动规范化改写而非拒绝。
- 文献复现（qec 环境，MILP 认证）：Lin–Pryadko Table III D6 码 `a=1+r⁴, b=1+sr⁴+r³+r⁴+sr²+r` → **[[24,8,3]] 精确复现**（d_exact=3），已用作 `lp_nonabelian` 内置种子；Table I SmallGroup(40,1)=C5⋊C8 码 → **[[80,8]] 复现，d_proxy=10**，其 GAP ID (40,1) 已入目录。记法换算：他们的 `s^ε r^k`（反射在前）= 本设计 `[(-k) mod n, ε]`（旋转在前）。
- BB72 [[72,12]]、BB144 [[144,12]] 基准 spec 在新 `bb` family 下复现；四 family 种子全部通过 sandbox → grammar → structural → Stim/DEM 全链路（dry-run funsearch + build-dem-smoke）。
- **已知未决对读项**：Lin–Pryadko Example 11 的 A4 例子（`a=1+x+y+x⁻¹yx, b=1+x+y+yx`，声称 [[24,5,3]]）在本实现的标准 2BGA 配对下穷举全部 `{e,x,y,·}×{e,x,y,·}` 乃至全部 w4×w4 子集组合均无 k=5、d≥3 的码（独立最小实现交叉验证一致，最优为 d=2）；判断为该例的词序/转录歧义，不影响上述两个成功复现所验证的构造链路。留待获取原文精确记法后核对。

## 13. 实施修订（2026-08-26，文档漂移校正）

本节按"代码 > docs/04 > 设计稿"的权威序，逐条记录本稿正文与当前实现之间已核实的偏差。正文保持定稿不动，以下各条以代码为准。

- **群目录名的正确拼法（§3.4 示例、§4.2 "命名" 列）**：`catalog.py` 的直积语法是 `^C(\d+)x(D\d+|A4|S4)$`（`construction/groups/catalog.py:51`），内层只接受 `D{n}`/`A4`/`S4`，且 `D_n` 的阶为 `2n`（§4.2 二面体行已声明 `D3 = S3`，直积行未沿用该换算）。写 seed spec 时**必须把文献记法换算成实现记法**：本文档 L101 与 L151 的 `C5xS3` 应写作 **`C5xD3`**（阶 30，GAP `(30,1)`，catalog.py:37），`C4xD10` 应写作 **`C4xD5`**（阶 40，GAP `(40,5)`，catalog.py:38）。两种误拼**都已不再是哑错**——报错本身点名目录拼法（2026-08-26 落地，回归 `tests/test_group_catalog_naming.py`）：`S3` 与 `C{m}xS3` 命中 `_LITERATURE_RE`（catalog.py:58、126-132），抛 `'C5xS3' is literature notation: S3 is the order-6 dihedral group, which this catalog indexes by rotation order as D3 — use 'C5xD3'`，把可直接复制回去的名字写进报错，末尾那条 `unknown non-abelian group`（catalog.py:133-135）只留给 `Q8` 这类真不在目录里的名字；`C4xD10` 仍是语法合法、被解析成**另一个群** C4 × D_10（阶 80、无 GAP ID）并由阶上限拒绝，但报错尾部现在追加 `; if this name came from the literature, its D10 is the order-10 dihedral group, which this catalog spells 'C4xD5'`（`_dihedral_index_hint`，catalog.py:68-79，由 `get_group` 的上限分支 :88-91 拼接），此前它只报 size 违规、属静默语义错配。该提示按证据设闸——只在折半读数确实是 `KNOWN_GAP_IDS` 条目时才出现，`C7xD9`（阶 126）因此只报大小、不臆造文献读数；该提示**挂在阶上限那条路径上**，因此上限一旦调整它就可能失效——`C4xD10`(阶 80) 若不再触发上限，提示随之消失、静默错配回归。为此另补了一道与上限无关的按名校验： `_LITERATURE_DIHEDRAL_SPELLINGS`（catalog.py）：`C4xD10` 这一个**文献确实印过**的拼法被列为保留名，由 `_construct` 在建群之前无条件拒绝并点名 `C4xD5`，与阶上限、与预算都无关。名单是显式枚举而非规则，因为规则分不清误抄与真需求（`C5xD6` 折半也落在目录项 `C5xD3` 上，但它本身是个正当的阶 60 群）。同时 `catalog_order("C4xD10")` 返回 `None` 而非字面阶 80：保留名不该被前置预算闸拿去算总量，否则 4×80=320 超过现网格任一预算，报出来的会是 `total_qubit_budget_violation`、`C4xD5` 指针照样丢失（回归 `tests/test_group_catalog_naming.py::test_reserved_spelling_outranks_the_budget_verdict`）。**走"拒绝 + 点名"而不是给目录加 `S3` 别名**，是因为 `construction/lp.py:62` 会把 `group.name` 回写进规范化 spec，一群两名会让同一个群以两种拼法进档案；这也正是目录既有的处置方式——`C2xD{n}` → `D{2n}`（catalog.py:218）、`C{m}:C2({k})` → `D{m}`（catalog.py:188）都是"拒绝 + 指向规范名"，模块 docstring 的 "Known-isomorphic presentations are rejected with a pointer to the canonical name" 现在把 S3 一并覆盖。同一改动还把索引约定写进了 `catalog_summary()`（"…the order-6 group often written S3 is D3 here and the order-10 one often written D10 is D5"），而该串原样渲染进 generator prompt（`construction/registry.py:213` → `funsearch.py:581`），这套换算不再只存在于人读的文档里；副作用是 generator prompt 文本随之变化而协议快照对此无感，记在 docs/04 §11 R1。§3.4 示例除群名外完全有效——`group` 改为 `C5xD3` 后可通过 `validate_lp_spec`。同理 §4.2 metacyclic 列的 `C5:C20(3)` 会被 twist 规范化改写为 `C5:C20(2)`（§12 已记该行为）。注：§4.2 设计要点、§10、§11 中带 `×`/`⋊` 的 `C5×S3`、`C4×D10` 是文献/数学记法，其附带 GAP ID 与实现一致，不属拼写问题。

- **群目录上限不随预算伸缩（§4.2 设计要点）**：预算反推本身确已落地——不等式 `(nA+mA)*(nB+mB)*|G| <= n_budget` 直接渲染进 generator prompt（`construction/registry.py:78`），并经 `memory/archive.py:101-107` → `verification/structural.py` 以 `total_qubit_budget_violation` 强制，故降低预算会按公式收窄可用 `|G|`。**未落地的只有「预算增大时目录自动扩展」**：目录另有一道固定常量 `MAX_GROUP_ORDER = 72`（`config.py`，即 288 总预算下双 1×1 的上界），不随 `n_budget` 伸缩，也不进 prompt，需与预算网格一起手工上调。因此 §4.2 括号中的 C5⋊C20（实现名 `C5:C20(2)`，阶 100）目前在任何预算下都被 `get_group` 拒绝，`C5xS4`(120)、`C13xA4`(156)、`C7xS4`(168)、`C13:C15(3)`(195) 同理——GAP ID 已在 `KNOWN_GAP_IDS` 里但取不出群。`get_group(name, *, max_order=…)` 的按调用覆盖口子已备好，暂无调用方。

  **为什么没顺手改成按预算推导**（2026-08-26 评估结论）：把上限接到预算上要求 `get_group` 拿得到 run 预算，而它经 `construction/lp.py::resolve_nonabelian_group` ← `validate_lp_spec` ← `plugin.validate(spec)` 到达，`plugin.validate` 只收 spec；`registry.py` 的契约行更是 import 期求值的模块级常量。也就是说整条群名/语法通路在设计上就与预算无关，接上去是一次真重构。而收益为零：最小 shape 因子是 4，故预算 N 下可用的最大 `|G|` 恰为 `N // 4`；`N ≤ 288` 时它总是 ≤ 72，即**上限在任何受支持的预算点上都不是绑定约束**（`shots_for` 表最高到 288），绑定的是预算本身。真要进入更大网格时再做，且届时应连同「是否写进 `build_protocol_snapshot()`」一起决定——搜索空间的上界目前不被协议冻结，改了不会被 `--resume` 拦住。

  同期确实落地的是**判定时机**：`catalog_order(name)` 从名字直接读出 `|G|`，上限在 `_construct` **之前**判（`construction/groups/catalog.py`）。原实现先造群再判阶，`D600` 要付约 28s 的 Cayley 表构造才被拒；现在 0.007ms。接受/拒绝的群集合逐名不变（242 个目录名差分验证零差异），回归见 `tests/test_group_catalog_naming.py::test_catalog_order_matches_what_construction_yields`。
- **`lp_shape_pin` 的强制点在 build 层，不在 grammar 层（§3.5 L131、§5 L180）**：`grammars/ga_seed.py` 的 `validate_ga_matrix` 没有 pin 参数；§5 所列其余各项（allowed_shapes、cell 非空/去重、指数范围、per-cell weight、supports 生成 G）确在 grammar 层。钉定的唯一强制点是 `construction/build.py:75-94` 的 `enforce_lp_shape_pin`，在 `build_compiled_code` 内 `plugin.validate()` 之后调用。故 §3.5 的"在 grammar 层报错"应读作"在 build 层报错"；报错文案 `lp_shape_pin: base matrix A must be 1x2, got …` 仍满足"明确指出钉定要求"。生效范围：搜索期由 `ArchiveMemory` 从 `config.lp_shape_pin` 注入（`funsearch.py:117`），structural 与 fast **两个门都传 pin**（`memory/archive.py:106`、`:141`），campaign 内每个候选都受钉定约束；不传 pin 的路径则不校验——`build_matrices`、slow 重验/仿真路径（`evaluation/pipeline.py`、`pool.py`、`worker.py`，与 build.py:36-38 的 docstring 一致），以及 `tools/precheck_draft.py`（违反钉定的草案能通过 precheck，到评估阶段才被拒；这是唯一会被人直接调用的漏口）。prompt 渲染侧按 §3.5 落地（`funsearch.py:951-959`）。

- **deformation 开关的实现形状与残留（§7）**：落地为单枚举 `RunConfig.deformation_mode`（`config.py:63-67` 的 `DeformationMode.NONE`/`SECTOR_H`，默认 `NONE`；CLI `--deformation-mode`），而非定稿描述的 `deformation.enabled` + `deformation.mode` 双字段——`none` 即关闭态，语义等价。构造层完整落地：`deformation/sector_h.py` 生成 H-mask，`construction/build.py:57,68,71` 写入 `CompiledCode` 并纳入 code hash（§7 L202 末句准确；§7 末句 L203 讲的是旧 btlp 路径，非 code hash）；"不进 genotype、不进 prompt"亦属实（`orchestrator/`、`llm/` 无 deformation 引用）。**但 ablation 目前无法端到端跑通**：`verification/circuit.py:156-159` 对任何非平凡 mask 抛 `NotImplementedError`（等待 02 §6.4 的 (check_type, sector) 门型表），且该异常被所有调用方吞掉——`evaluation/pipeline.py:199-207` 记为 `circuit_error` 后 `continue`，`cli.py:770-774` 记为 `status:"error"` 并返回退出码 0。因此 `sector_h` 跑批**不会显式失败**，而是烧完预算产出无 DEM、无 LER 的结果，开启前须知。

- **非阿贝尔断言的实际形式（§3.4 L109）**：`construction/lp.py:56` 的断言是 `family == "lp_nonabelian" and group.is_abelian` → 拒绝，即检查**目录群 G** 是否阿贝尔，而非对 `⟨supp(A), supp(B)⟩` 的独立断言。当前与生成性检查合起来结论相同，但 §3.4 声称的"防止后续放松生成性约束时静默失去该保证"并不成立——一旦去掉 `require_supports_generate`，非阿贝尔 G 上落在阿贝尔子群里的 support 会静默通过。若要恢复该保证，需改为对 `group.subgroup_closure(support)` 判非阿贝尔。

- **实际目录树比 §8 多两个模块**：`construction/groups/base.py`（§4.3 的 pc 正规形、Cayley 表与子群闭包的公共实现，`abelian.py` 与 `catalog.py` 共用）与 `construction/structure.py`（结构坐标描述符，供 `memory/coverage.py`、`memory/ledger.py`、`workflow/graph.py` 使用，本稿定稿后新增）。§8 列出的其余文件与实现一一对应；`deformation/` 的文件名（`api.py`、`sector_h.py`）亦与 §8 一致，但 §8 给 `api.py` 标注的签名 `apply(compiled, config) -> CompiledCode` 与实现不符——实际入口是 `resolve_h_mask(mode, *, sector_sizes, n_data) -> list[int]`（`deformation/api.py:10-23`，也是 `deformation/__init__.py` 的唯一导出），返回 H-mask 列表而非 CompiledCode，模块本身不碰 CompiledCode；写回由构造层完成（`construction/build.py:57` 取 mask、`:68` 落入 `CompiledCode.h_mask`）。

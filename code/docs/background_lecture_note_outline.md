# OmniQEC Background Lecture Note：大纲定稿

- 状态：**大纲定稿 v5**（2026-08-28，讨论五轮收敛）
- 定位：论文 background 展开成 lecture note，面向 researcher。结构 = 该 genre 论文的标准前件展开：**短漏斗 Introduction + 两个分层 Background 章**（成对背景章的已发表先例：Nautrup et al., Quantum 3, 215 (2019)）。
- 配比：Part 0 ≈ 5%，**Part I (Quantum) ≈ 60%**，Part II (LLM) ≈ 35%。
- 依据：两轮文献调研（§6 写作约定的全部来源均已核对原文）+ `omniqec_agentic/docs/01–04` 设计稿。
- 可拆卸原则（Olle et al., npj QI 2024 的教训）：两个 Background 章写成可拆卸单元——将来论文投稿时可整体降为 Methods/SI，正文不动。

---

## 0. 总主线（一句话）

> LP 是一个巨大的构造母空间；qLDPC 的文献史就是人类靠洞察在这张地图上一刀一刀切出结构化区域（HGP、2BGA、BB、Mitten……）、再在小区域里人工找好码的历史；OmniQEC 把「往哪切、在切出的区域里怎么搜」交给 agent，用冻结的电路级测量协议裁决输赢。

---

## 1. Part 0 · Introduction（漏斗，1–1.5 页）

全部「为什么」住在这里，每步 1–3 句，不进 Background 章：

1. **Physics-stakes 开场**（Olle 语式）：QEC 是有用量子计算的门；硬件里程碑让「用哪个码」成为活的工程问题；
2. **问题 + 硬度**：好码住在 LP 母空间的稀疏角落；文献史 = 人类逐区域切分 + 人工搜索（组合爆炸、逐平台适配两个论证各一句）；
3. **Forcing gap → LLM 进场**（braiding 句）：构造可以廉价指定、精确验证，而构造→性能的映射高度非平凡——恰是 LLM 引导搜索的适用区；
4. **先例一句话分类**：FunSearch 谱系（program search + evaluator as truth）；QEC 已有两例（IBM qcode-discovery、Liu–Marquardt SCE）但均止步 code-capacity；
5. **贡献句 + 正确性前置**：agentic lab-in-the-loop，电路级协议冻结、verifier 写进搜索定义本身（而非事后验证）；
6. **路线图**：两个 Background 章的地图。

---

## 2. Part I · Quantum Background（~60%，主体）

### I.1 从 qubit 到错误的必然性（短引入）

- qubit / 门 / 测量一段带过（默认已知，只作叙事起点）；
- 噪声与退相干：物理门 ~10⁻³ vs 算法需要 ~10⁻¹⁰，一句话点出差距 → 纠错必经之路。

### I.2 Stabilizer 与 CSS 形式主义（代数先行，B&E house style）

- 从经典纠错直觉（repetition、parity check）→ 量子三障碍（no-cloning、测量塌缩、错误连续）；
- stabilizer formalism 一次解决三者：syndrome 测量离散化错误、不触碰逻辑信息；`[[n,k,d]]`、distance、逻辑算符、Tanner 图在叙事需要处自然出现（不做记号页）；
- CSS：`H_X H_Z^T = 0`，X-check 抓相位翻转、Z-check 抓比特翻转；
- **surface code 作为 illustration 出现在本节末**（不是入口），带一句局限注脚（k=1、rate→0；BPT 界注脚，兼作 Q = kd²/n 的出处）。

### I.3 Code family 的完整数学构造（核心，5–6 页）

每个构造给**显式矩阵形式 + 参数推导 + 分块拆解**；「为什么选这几个 / 优势」不在此论证。

#### I.3.0 铺垫

- family 概念：构造模板 + 自由参数；**LDPC 是 family 的性质不是单个码的性质**（B&E 原文引用）；
- 两条历史支流一小段：two-block 支流（bicycle '04 → GB '13 → BB '24）；product 支流（HGP '09 → LP '20/'21）；Lin–Pryadko '23 汇合——two-block 被识别为「最小 LP」；
- 一句话注脚（事后视角）：历史顺序与逻辑包含相反（HGP 比 LP 早 11 年），是 LP 出现后此前 family 才被回溯性识别为其子区域。

#### I.3.1 积构造（HGP）

- 显式分块形式：H₁ (m₁×n₁)、H₂ (m₂×n₂)，
  `H_X = [H₁⊗I | I⊗H₂ᵀ]`，`H_Z = [I⊗H₂ | H₁ᵀ⊗I]`；
- CSS 条件一行验证（Kronecker 混合积律）；
- 参数推导：`n = n₁n₂ + m₁m₂`，k 经 transpose code 计数，`d = min(d₁,d₂)`（证明 sketch）；
- Tanner 图 = 图积；worked example：**toric = 两个 repetition code 的 HGP**，显式写出。

#### I.3.2 Lift（LP）

- 群代数 F₂[G]、正则表示；lift = 把 F₂ entry 换成 F₂[G] entry，每个 entry 膨胀为 |G|×|G| 置换矩阵块；
- LP(A,B) 完整形式（A: m_A×n_A，B: m_B×n_B over F₂[G]）；
- 参数公式：`N_data = (n_A n_B + m_A m_B)·|G|`，`N_checks = (m_A n_B + n_A m_B)·|G|`；
- HGP = G 平凡的退化情形（同一公式一键还原）。

#### I.3.3 2BGA（1×1 行）

- `A = L(a)` 左正则、`B = R(b)` 右正则，a, b ∈ F₂[G]；
- **左乘右乘对易的两行证明 → CSS 条件由构造免费获得**；
- `n = 2|G|`，k 来自 rank 亏损。

#### I.3.4 BB（阿贝尔双因子特化）

- G = Z_ℓ × Z_m，元素 = 单项式 x^a y^b，A、B = 稀疏双变量多项式（"bivariate" 名字的由来：阿贝尔 2BGA 按循环因子数分层——单因子 GB，双因子 BB）；
- `H_X = [A|B]`，`H_Z = [Bᵀ|Aᵀ]`，`n = 2ℓm`；
- worked example：gross code `A = x³+y+y²`、`B = y³+x+x²` → [[144,12,12]]，从多项式到矩阵到参数走通一遍。

#### I.3.5 非阿贝尔情形

- 可解群的 pc 正规形指数元组编码（每个元素唯一写成 g₁^e₁···g_k^e_k）；
- 群目录四支：二面体 / 交错对称 / metacyclic / 直积；
- Mitten 1×2 形式 `A = [a₀ a₁]`；**阿贝尔 1×2 距离封顶定理 sketch + 非阿贝尔为何解除封顶**——谱系收束点；
- 渐近好码需要非阿贝尔 + expansion 一句话（PK '21），与上一条合成「非阿贝尔群是资源」两票。

#### I.3.6 「如何拆开」：统一的双 sector 分块结构 + LP 疆域地图

- 四个构造的 H_X/H_Z 天然携带两块边界（积构造的通性）：bb 两个 ℓm 块；hgp `n₁n₂ | m₁m₂`；LP `(n_A n_B)|G| | (m_A m_B)|G|`；
- **该拆解正是 I.4 中 SEC 的 partition 输入**（四 family 的天然两块结构使我们跳过一般 CSS 码的 partition 搜索）——I.3 与 I.4 在数学上焊接；
- **LP 疆域地图**（Part I 的主图，shape × group 二维格子）：

| shape \ G | 平凡群 {e} | 循环 C_ℓ | 阿贝尔 | 非阿贝尔 |
|---|---|---|---|---|
| 任意 m×n | **HGP**（TZ '09） | quasi-cyclic LP（PK '20） | abelian LP | **general LP**（PK '21：+expansion → 渐近好码） |
| 1×2 | — | | 距离被 base matrix check weight 封顶 | **Mitten codes**（'26，rate 1/5、w=9） |
| **1×1**（two-block 行 = 2BGA） | （退化） | **GB** | **BB**（双因子） | 非阿贝尔 2BGA（L–P census, w≤8） |

- 地图讲三点：HGP = 钉死群轴的整列、2BGA = 钉死形状轴的整行（两轴独立特化，有重叠、不穷尽——Mitten 格点两边都不属于）；toric 坐在多重交点上（= 两个 repetition 的 HGP = A=1+x, B=1+y 的 BB）作自洽检验；BPT 界写在地图角落；
- **四 grammar 落位一段说明**（不论证）：`bb` = 1×1 × Z_ℓ×Z_m；`hgp` = 任意形状 × 平凡群；`lp_abelian` / `lp_nonabelian` = ≤2×2 小 seed × 阿贝尔/非阿贝尔目录；受控对比各一句（bb vs lp_abelian 隔离形状轴、lp_abelian vs lp_nonabelian 隔离群轴、hgp vs lp 隔离 lift）；四区域共享同一预算、w_max ≤ 9、噪声、SEC、decoder——grammar 比较而非算力比较。

### I.4 Syndrome Extraction Circuit（完整一节，3–4 页）

#### I.4.1 为什么需要电路层

- 测一个 weight-w stabilizer = 1 个 ancilla + w 个 CNOT + 读出；每轮所有 check 并行调度；
- code-capacity vs circuit-level：错误发生在电路每个位置、且经 CNOT 传播。

#### I.4.2 调度非平凡：hook 错误与电路距离

- ancilla 单错经 CNOT 序传播成数据比特多体错 → **d_circ ≤ d**，坏调度灾难性压低它；
- 两个问题引出：怎么排得浅（深度），怎么排得不伤距离（有效性/排名）。

#### I.4.3 深度：边着色

- 每轮调度 = Tanner 图边着色；二部图 König 定理（色数 = Δ）→ 最小深度由最大度决定；
- **check weight 在这里兑现成物理代价**——w_max 约束的真正来源。

#### I.4.4 LRC 构造（SBB）

- 输入 = I.3.6 的双 sector 拆解：H_X = [L_X|R_X]、H_Z = [L_Z|R_Z]，四个子图各自最小着色；
- 每轮两个窗口（L_X ∥ R_Z，然后 L_Z ∥ R_X）+ X/Z ancilla 的 staggered 制备/测量；
- **有效性由构造保证**（staggering ⇒ 同轮非交错 ⇒ detector 确定），不靠事后修复；
- 深度公式 `t = t₁+t₂+2`；各 family 预期：bb `w_A+w_B+2`、hgp `δ₁+δ₂+2`、lp `δ_A+δ_B+2`；
- 同深度候选的排名式选择一段（Δmin → τ_A → W；schedule 可以合法地灾难——排名的价值所在）；`d_circ_ext` 证据通道一句带过。

#### I.4.5 测量闭环：噪声、解码、度量

- 噪声 profile（uniform / si1000 / helios）各一句；decoder（beam B=64 / BP+LSD）各一句；
- 纵轴：**E5/E6 per-logical-qubit-per-round LER**（引组内 decoder 论文 arXiv:2606.27119 SI E.1，跨 k、跨轮数公平——组内两篇工作口径同源）；
- 「赢」的定义：同协议 beats-BB 且 CI 分离 → protocol 冻结清单亮出，交棒 Part II。

---

## 3. Part II · LLM Background（~35%）

### II.1 从 LLM 到 agent（半页）

- 预训练先验 / in-context / 工具调用一段带过；
- **tool vs agent 区分**（Melnikov 移植：代码补全工具 vs 验证环里的自主发现 agent）；
- 两个缺陷埋点（后文机制化回收）：会幻觉（说的话不可信）、无持久学习（上下文之外什么都不留下）。

### II.2 LLM 驱动的发现循环（1 页，讲机制不做比较）

- **Program search**：搜「生成对象的程序」而非对象本身；LLM 作为 mutation operator；**正确性是搜索定义的一部分**——evaluator 执行、打分、丢弃，进入基因池的只有已验证程序；
- 进化记忆：archive、MAP-Elites / islands、quality-diversity；
- agentic / lab-in-the-loop 闭环一段（Robin / Co-Scientist 各一句）；
- QEC 先例一句带过（IBM、SCE，均 code-capacity）；
- gloss 纪律：架构名当专有名词，不展开 ML 内部机制。

### II.3 OmniQEC 架构（核心，配双图）

- 双图分工（FunSearch 模式）：**translation 图**（genotype spec ↔ 编译出的码）+ **system 图**（双环回路）；
- **两层分权公理**：policy（LLM 决定测谁、花多少、往哪探）/ protocol（冻结的测量：噪声、SEC、decoder、统计）——「agentic 的是决策，可复放的是测量」；
- 幻觉的机制化处理（回收埋点①）：工具三档 scope（read/draft/policy）、schema 强制引用、agent 的话永不成为记录、report 从产物文件渲染；
- **双角色**（刻意不做 multi-agent 编排）：Orchestrator（每轮一个 TurnPlan、带来源哈希的上下文块）+ Generator（每代一批 `construct(rng)` 程序）；
- fast/slow 双环：sandbox → 结构检查 → Q proxy → MAP-Elites → 晋级 → SEC → 解码 → plpr + anchor-p racing → reconcile；
- 预算四本账（tokens / shots / 墙钟 / turns）。

### II.4 学习层（1–1.5 页）

- 立论（回收埋点②）：裸 in-context 反馈不构成学习（Gupta et al. 一句话）→ **学习必须外化为经过验证的工件**；
- 五组件收为一表，重点展开两个：**insight ledger**（agent 只有提案权）+ **前瞻回测器**（只用提案后的证据判定，反 double-dipping；curator 是确定性的）；其余三项各一句：UCB1 bandit（建议非命令）、两层指纹 + 已知码表（只做重发现判定、不注入先验）、结构覆盖度（"没探过" ≠ "探过但输了"）+ 手册/递减窗口反思；
- **Prompt 三原则**：坏写死、目标极简、路径留白——路径写进 prompt 就是先验污染，那正是要留给搜索的空间。

### II.5 溯源与可复现（半页–1 页，开源向）

- prompt-as-code（内容哈希入协议快照）、协议快照 + resume 校验、全 transcript 落盘、确定性种子 / shard 预注册、blind-baseline 纪律（prompt 永不提 BB）、成本双口径核算（token + 实际计费金额）。

### II.6 局限与诚实性（半页）

- evaluator 可被 game（自家 d_proxy 实例）；rediscovery 的置换等价**上界**口径；序贯停止偏差只披露不校正；LLM 随机性以 transcript + 冻结协议 + 全链路种子保证可审计；
- 留一句 **retroactive vindication** 槽位（"模型找到的解具有人类基线没有的性质 X"——发现物出来后回填）。

---

## 4. 记号约定

- LP/2BGA 部分跟 **Lin–Pryadko / Mitten 约定**（也是代码实现的约定）；
- BB 部分用 **Bravyi 多项式记号**，给一句与 LP 记号的对应关系；
- 群元素编码用 pc 正规形指数元组（对外接口），与 GAP `Elements(G)` 索引的换算一句注脚（对读文献码表用）。

## 5. 文献锚点（均已核对原文）

| 用途 | 文献 |
|---|---|
| qLDPC 综述 / LDPC-family 定义 | Breuckmann & Eberhardt, PRX Quantum 2, 040101 (2021), arXiv:2103.06309 |
| HGP | Tillich & Zémor, arXiv:0903.0566 |
| LP / 渐近好码 | Panteleev & Kalachev, arXiv:2012.04068; arXiv:2111.03654 |
| BB / gross code | Bravyi et al., Nature 627, 778 (2024), arXiv:2308.07915 |
| GB | Kovalev & Pryadko, arXiv:1212.6703；bicycle: MacKay et al. (2004) |
| 2BGA census | Lin & Pryadko, arXiv:2306.16400 |
| Mitten codes | arXiv:2607.28795 |
| SEC / LRC | Strikis–Browne–Beverland, arXiv:2603.05481；coloration: Tremblay–Delfosse–Beverland, PRL 129, 050504 (2022) |
| decoder | Beam: arXiv:2512.07057 (PRXQ 2026)；BP+LSD: Hillmann et al., Nat Commun 16, 8214 (2025) |
| plpr 度量（E5/E6） | 组内 decoder 论文 arXiv:2606.27119, SI E.1 |
| BPT 界 | Bravyi–Poulin–Terhal（2D 局域 kd² ≤ O(n)） |
| LLM 发现谱系 | FunSearch, Nature 625, 468 (2024)；AlphaEvolve, arXiv:2506.13131 |
| agentic / lab-in-the-loop | Co-Scientist, Nature (2026)；Robin, Nature (2026) |
| LLM-for-QEC 先例 | IBM qcode-discovery, arXiv:2606.02418；Liu & Marquardt SCE, arXiv:2606.24808 |
| 学习外化立论 | Gupta, Hartford & Liu, Findings of EMNLP 2025, arXiv:2509.21403；ACE, arXiv:2510.04618 |
| RL-for-QEC（genre 先例） | Nautrup et al., Quantum 3, 215 (2019)；Olle et al., npj QI 10 (2024) |

## 6. 写作约定（来自两轮 genre 调研，起草时执行）

1. **领域开场，方法永不开场**；方法进场前一句制造 forcing gap（领域自己的工具走到头）；
2. **gloss 不对称**：领域概念逐个内联解释（同位语式一句话），ML 架构名当专有名词不定义；
3. **「ML 在拿到方程之前，领域先拿到图」**：Part I 主图 = LP 疆域地图，Part II 才出现 loop 图；
4. **正确性写进搜索定义**，不作为事后 validation 段落；
5. 先例工作**只分类不争论**，两条引用谱系分段隔离；
6. 数字只花两处：问题普遍性 + 搜索空间大小；intro/背景章不出现 ML 性能指标；
7. 每个概念在**叙事需要处**自然出现（PK 的 point-of-use 规则），不做集中的记号/定义页；
8. surface code 是 illustration 不是入口（B&E house style：代数先行）。

## 7. 版本沿革（讨论决策记录）

| 版本 | 关键决策 |
|---|---|
| v1 | 初版：按时间顺序的谱系 + 论证式 background |
| v2 | 用户修正：以 LP 为纲的包含关系做主结构；修正为二维格子（shape × group，非树——Mitten 格点两边不属）；BB = 阿贝尔 2BGA 双因子层；「母空间/切区域」语言确立 |
| v3 | 用户修正：删独立动机页与记号页；两部分改为对称教学递进（basics → 领域 → 我们的对象） |
| v4 | genre 调研裁决：教学递进 = PRXQ Background 章惯例；恢复短漏斗 Introduction（Part 0）承载全部「为什么」；surface code 位置、point-of-use 归属定案 |
| v5 | 用户修正：Quantum 为主体（60%）；I.3 升格为逐构造完整数学展开（显式矩阵 + 参数推导 + worked examples）；I.4 升格为完整 SEC 章；I.3.6 双 sector 拆解与 I.4.4 partition 数学焊接；Part II 收缩 |

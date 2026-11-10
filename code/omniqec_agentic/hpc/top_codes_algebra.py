"""合并 CSV 里 top 码的代数形式 + family 归属复核（中文报告）。

用法:
    python -m omniqec_agentic.hpc.top_codes_algebra <放 fig2_all_best_*.csv 的目录> \
        [--budgets 60 98] [--site v36] [--out <目录>] [--top 5]

对每个 (arm, budget) 取 fig2_all_best_<model>.csv 的 BB 基线行（rank 0，若有）和前 N 名，
对每个码：
  1. 用 campaign 自己的 family 构造器（construction.build.build_matrices）按 `spec` 列
     重建 HX/HZ；
  2. 证明它就是被测的码：compiled_code_hash(HX,HZ)[:8] == CSV 的 code_hash8——这个
     哈希是 campaign 在集群上对实际送去解码的矩阵算出来的，所以不需要再从集群下载任何产物；
  3. 在 GF(2) 上重算 n 与 k = n - rank HX - rank HZ，与 CSV 的 n_data / k 对照；
  4. 把 spec 渲染成 F2[G] 上的多项式记法，并点名结构类别（ℓ=1 的 BB 实为单变量 GB 码、
     两个环码的 HGP 是 toric 码、1×1 的 lifted product 是 2BGA 码），便于肉眼判断 family。

输出 <out>/top_codes_<tag>.md（给人看）、<out>/top_codes_<tag>.json（机器读）和
<out>/matrices/<arm>_b<budget>_r<rank>_<hash8>.npz（hx、hz，uint8）。约定写在 md 开头。
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.construction.build import build_matrices, compiled_code_hash
from omniqec_agentic.construction.lp import resolve_abelian_group, resolve_nonabelian_group
from omniqec_agentic.construction.registry import get_family
from omniqec_agentic.deformation import resolve_h_mask

_MODEL_RE = re.compile(r"^fig2_all_best_(?P<model>.+)\.csv$")


# ----------------------------------------------------------------- GF(2) rank
def gf2_rank(matrix: np.ndarray) -> int:
    m = (np.asarray(matrix, dtype=np.uint8) % 2).copy()
    rows, cols = m.shape
    rank = 0
    for c in range(cols):
        pivot = next((r for r in range(rank, rows) if m[r, c]), None)
        if pivot is None:
            continue
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        for r in np.nonzero(m[:, c])[0]:
            if r != rank:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


# ------------------------------------------------------------- 渲染
def _mono(names: list[str], exps: list[int]) -> str:
    parts = []
    for name, e in zip(names, exps):
        if e == 0:
            continue
        parts.append(name if e == 1 else f"{name}^{e}")
    return " ".join(parts) if parts else "1"


def _poly(names: list[str], terms: list[list[int]]) -> str:
    return " + ".join(_mono(names, [int(v) for v in t]) for t in terms) if terms else "0"


def _ga_matrix(names: list[str], cells: list) -> str:
    rows = ["  ".join(f"({_poly(names, cell)})" if len(cell) != 1 else _poly(names, cell)
                      for cell in row) for row in cells]
    if len(rows) == 1:
        return "[ " + rows[0] + " ]"
    return "[ " + " ;\n          ".join(rows) + " ]"


def _bin_matrix(rows: list[list[int]]) -> str:
    return "\n".join("      " + " ".join(str(int(v)) for v in row) for row in rows)


def _generator_names(group_name: str, kind: str, n_gen: int) -> tuple[list[str], str]:
    """指数元组各分量对应的变量名 + 一行群的表现（presentation）。"""

    if kind == "abelian":
        names = ["x", "y", "z", "w", "u", "v"][:n_gen]
        return names, "阿贝尔群，生成元两两交换；元组第 i 个分量是第 i 个生成元的指数"
    if re.match(r"^D\d+$", group_name):
        n = int(group_name[1:])
        return ["r", "s"], f"D{n}：r^{n} = s^2 = 1，s r s = r^-1（|G| = {2 * n}）；元组 [i,j] = r^i s^j"
    m = re.match(r"^C(\d+):C(\d+)\((\d+)\)$", group_name)
    if m:
        p, q, k = (int(v) for v in m.groups())
        return ["r", "s"], f"C{p}:C{q}({k})：r^{p} = s^{q} = 1，s^-1 r s = r^{k}（|G| = {p * q}）；元组 [i,j] = r^i s^j"
    m = re.match(r"^C(\d+)x(D\d+|A4|S4)$", group_name)
    if m:
        p, inner = int(m.group(1)), m.group(2)
        if inner.startswith("D"):
            n = int(inner[1:])
            return ["c", "r", "s"], (f"C{p} × D{n}：c^{p} = 1 且 c 中心；r^{n} = s^2 = 1，s r s = r^-1"
                                     f"（|G| = {2 * n * p}）；元组 [a,i,j] = c^a r^i s^j")
        if inner == "A4":
            return ["c", "a", "u", "v"], f"C{p} × A4：c 中心、阶 {p}；a 阶 3，u、v 是 Klein 四元群；元组 [e,i,j,k] = c^e a^i u^j v^k"
        return ["c", "t", "a", "u", "v"], f"C{p} × S4：c 中心、阶 {p}；t 对换，a 阶 3，u、v 是 V4；元组 = c^e t^i a^j u^k v^l"
    if group_name == "A4":
        return ["a", "u", "v"], "A4：a 阶 3，u、v 是 Klein 四元群；元组 [i,j,k] = a^i u^j v^k"
    if group_name == "S4":
        return ["t", "a", "u", "v"], "S4：t 对换，a 阶 3，u、v 是 V4；元组 = t^i a^j u^k v^l"
    return [f"g{i + 1}" for i in range(n_gen)], "生成元 g1..gn 按 catalog 顺序"


def _is_ring_seed(h: np.ndarray) -> bool:
    """方阵、每行每列重量 2、且各行当作边构成单一一个圈（循环重复码）。"""

    r, n = h.shape
    if r != n or not np.all(h.sum(1) == 2) or not np.all(h.sum(0) == 2):
        return False
    adj: dict[int, set[int]] = {c: set() for c in range(n)}
    for row in h:
        a, b = np.nonzero(row)[0]
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    seen, stack = set(), [0]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(adj[v])
    return len(seen) == n


# 非阿贝尔目录群的 GAP SmallGroup 编号（文献用它做群的身份标识，Mitten 表 VII、Lin–Pryadko 表 I）。
# 先取目录里文献钉过的，再补这几阶上由结构描述唯一确定的群。
_EXTRA_GAP_IDS = {"D12": (24, 6), "C3xD4": (24, 10), "C5:C4(2)": (20, 3), "C7:C3(2)": (21, 1)}


def _gap_id(group) -> str:
    if group.gap_id is not None:
        return f"SmallGroup({group.gap_id[0]},{group.gap_id[1]})"
    gid = _EXTRA_GAP_IDS.get(group.name)
    return f"SmallGroup({gid[0]},{gid[1]})" if gid else "（未标注）"


def _invariant_factors(factors: list[int]) -> list[int]:
    from omniqec_agentic.construction.groups.abelian import invariant_factors
    return [int(v) for v in invariant_factors(tuple(int(q) for q in factors))]


def _mitten_conjugation(group_name: str) -> str:
    """Mitten 表 VII 的写法：φ_s(r) = s r s^-1 = r^c，c = k^-1 mod m。"""
    m = re.match(r"^C(\d+):C(\d+)\((\d+)\)$", group_name)
    if m:
        p, q, k = (int(v) for v in m.groups())
        return f"⟨r, s | r^{p} = s^{q} = e⟩，φ_s(r) = s r s^-1 = r^{pow(k, -1, p)}"
    m = re.match(r"^D(\d+)$", group_name)
    if m:
        return f"⟨r, s | r^{int(m.group(1))} = s^2 = e⟩，φ_s(r) = s r s^-1 = r^-1"
    m = re.match(r"^C(\d+)x(D\d+|A4|S4)$", group_name)
    if m:
        return "直积，因子名已唯一确定群，无需扭曲关系"
    return ""


def _elementwise_inverse(cells: list, group) -> list:
    return [[[list(group.elements[group.inverse[group.element_index(e)]]) for e in cell] for cell in row] for row in cells]


def describe(spec: dict[str, Any]) -> dict[str, Any]:
    """从 VALIDATED spec 得到代数渲染 + 结构类别。"""

    family = spec["family"]
    out: dict[str, Any] = {"family": family}
    if family == "bb":
        ell, m = int(spec["ell"]), int(spec["m"])
        names = ["x", "y"]
        a, b = _poly(names, spec["A"]), _poly(names, spec["B"])
        out.update({
            "ring": f"F2[x,y] / (x^{ell} - 1, y^{m} - 1)（ℓ={ell}, m={m}，n = 2ℓm = {2 * ell * m}）",
            "A": a, "B": b,
            "weights": f"w_A={len(spec['A'])}, w_B={len(spec['B'])}",
            "checks": "HX = [A | B]，HZ = [B^T | A^T]",
        })
        if ell == 1:
            out["structural_class"] = (f"单变量：ℓ=1 使 x 平凡，实为循环群 C{m} 上的 generalized bicycle（GB）码"
                                       f"（也可写成 C{m} 上 1×1 的 lp_abelian）")
        else:
            out["structural_class"] = f"C{ell}×C{m} 上的双变量 bicycle（BB）码"
        out["algebra"] = f"A = {a}\n    B = {b}\n    环：{out['ring']}"
    elif family == "hgp":
        h1 = np.asarray(spec["seeds"]["H1"], dtype=np.uint8)
        h2 = np.asarray(spec["seeds"]["H2"], dtype=np.uint8)
        r1, n1 = h1.shape
        r2, n2 = h2.shape
        ring1, ring2 = _is_ring_seed(h1), _is_ring_seed(h2)
        out.update({
            "H1_shape": f"{r1}x{n1}", "H2_shape": f"{r2}x{n2}",
            "checks": "HX = [H1 ⊗ I_n2 | I_r1 ⊗ H2^T]，HZ = [I_n1 ⊗ H2 | H1^T ⊗ I_r2]",
            "n": f"n1·n2 + r1·r2 = {n1 * n2} + {r1 * r2} = {n1 * n2 + r1 * r2}",
            "seed_class": {"H1": "环码（循环重复码）" if ring1 else "一般种子",
                           "H2": "环码（循环重复码）" if ring2 else "一般种子"},
        })
        if ring1 and ring2:
            out["structural_class"] = (f"toric 码：长度 {n1} 与 {n2} 的两个环码的 HGP = {n1}×{n2} 环面上的 toric code，"
                                       f"[[{2 * n1 * n2}, 2, {min(n1, n2)}]]")
        else:
            out["structural_class"] = "一般经典种子的 hypergraph product"
        out["algebra"] = f"H1（{r1}x{n1}）=\n{_bin_matrix(spec['seeds']['H1'])}\n    H2（{r2}x{n2}）=\n{_bin_matrix(spec['seeds']['H2'])}"
    else:  # lp_abelian / lp_nonabelian
        group = (resolve_abelian_group if family == "lp_abelian" else resolve_nonabelian_group)(spec)
        names, presentation = _generator_names(group.name, group.kind, len(group.generator_orders))
        a_cells, b_cells = spec["A"], spec["B"]
        m_a, n_a, m_b, n_b = len(a_cells), len(a_cells[0]), len(b_cells), len(b_cells[0])
        entries = [tuple(e) for row in a_cells + b_cells for cell in row for e in cell]
        idx = [group.element_index(e) for e in entries]
        out.update({
            "group": group.name, "group_order": group.order, "group_is_abelian": bool(group.is_abelian),
            "presentation": presentation,
            "generator_orders": list(group.generator_orders),
            "A_shape": f"{m_a}x{n_a}", "B_shape": f"{m_b}x{n_b}",
            "supports_generate_G": bool(group.generates(idx)),
            "checks": "HX = [L(A ⊗ I_nB) | R(I_mA ⊗ B*)]，HZ = [R(I_nA ⊗ B) | L(A* ⊗ I_mB)]，* = 转置 + 反极（g → g^-1）",
            "n": f"(nA·nB + mA·mB)·|G| = ({n_a * n_b} + {m_a * m_b})·{group.order} = {(n_a * n_b + m_a * m_b) * group.order}",
        })
        if m_a == n_a == m_b == n_b == 1:
            out["structural_class"] = (f"1×1 lifted product = 群 {group.name} 上的两块群代数（2BGA / generalized bicycle）码："
                                       f"HX = [L(a) | R(b*)]，HZ = [R(b) | L(a*)]，n = 2|G| = {2 * group.order}")
        else:
            out["structural_class"] = f"群 {group.name} 上的 lifted product，基矩阵 A 为 {m_a}×{n_a}、B 为 {m_b}×{n_b}"
        if family == "lp_nonabelian" and group.is_abelian:
            out["structural_class"] += "  ！！！该群是阿贝尔群"
        if group.kind == "abelian":
            inv = _invariant_factors(list(group.generator_orders))
            out["invariant_factors"] = inv
            out["group_identity"] = "不变因子 " + "×".join(f"C{q}" for q in inv) + ("（循环群）" if len(inv) == 1 else f"（{len(inv)} 个生成元）")
        else:
            out["gap_id"] = _gap_id(group)
            out["mitten_presentation"] = _mitten_conjugation(group.name)
            out["group_identity"] = f"GAP {out['gap_id']}"
        b_bar = _ga_matrix(names, _elementwise_inverse(b_cells, group))
        out["B_bar_mitten"] = b_bar
        out["algebra"] = (f"G = {group.name}：{presentation}\n    A = {_ga_matrix(names, a_cells)}\n"
                          f"    B = {_ga_matrix(names, b_cells)}\n"
                          f"    按 Mitten 论文 Def.8/10 的约定（R(g): h ↦ h g^-1）同一个码写作 LP(A, B̄)，B̄ = B 逐元素取逆：\n"
                          f"    B̄ = {b_bar}")
    return out


# ------------------------------------------------------------------ 行
def load_rows(csv_dir: Path, budgets: set[str], site: str, top: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(csv_dir.glob("fig2_all_best_*.csv")):
        model = _MODEL_RE.match(path.name).group("model")
        for r in csv.DictReader(open(path, newline="")):
            if r["budget"] in budgets and r["site"] == site and int(r["rank"]) <= top:
                r["_model_file"] = model
                rows.append(r)
    rows.sort(key=lambda r: (r["arm"], int(r["budget"]), int(r["rank"])))
    return rows


def verify(row: dict[str, str]) -> dict[str, Any]:
    spec = json.loads(row["spec"])
    plugin = get_family(spec["family"])
    validated = plugin.validate(spec)
    hx, hz = build_matrices(spec)
    n = int(hx.shape[1])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    css_ok = int((hx.astype(int) @ hz.T.astype(int) % 2).sum()) == 0
    w_x, w_z = int(hx.sum(1).max()), int(hz.sum(1).max())
    q_deg = int((hx.sum(0) + hz.sum(0)).max())
    h8 = compiled_code_hash(hx, hz)[:8]
    hash_match = h8 == row["code_hash8"]
    hash_note = ""
    if not hash_match:  # campaign 若开了 sector_h 形变，哈希会带 H 掩码
        sectors = [int(v) for v in plugin.sector_sizes(validated)]
        mask = resolve_h_mask("sector_h", sector_sizes=sectors, n_data=n)
        alt = compiled_code_hash(hx, hz, h_mask=mask)[:8]
        if alt == row["code_hash8"]:
            hash_match, h8, hash_note = True, alt, "（含 sector_h H 掩码）"
    return {
        "validated_spec": validated,
        "hx": hx, "hz": hz,
        "n": n, "k": k, "css_orthogonal": css_ok, "w_x": w_x, "w_z": w_z, "qubit_degree": q_deg,
        "hash8_rebuilt": h8 + hash_note, "hash_match": hash_match,
        "n_match": n == int(row["n_data"]), "k_match": k == int(row["k"]),
        "describe": describe(validated),
    }


_CONVENTIONS = """## 验证链（为什么不必再从集群下载任何东西）

CSV 的 `code_hash8` 是 campaign 在集群上、对实际送去解码的那对 HX/HZ 矩阵算出的 SHA-256
（`construction.build.compiled_code_hash`，取前 8 位十六进制）；`spec` 列是导出 CSV 时按同一哈希从该码
的产物里取出来的构造参数。本报告在本机按 `spec` 用同一套构造器重建 HX/HZ 并重算哈希：哈希相同就说明
"下面写出的代数形式"与"集群上被测的矩阵"逐字节一致，与 `spec` 是否被抄错、产物是否被同名覆盖都无关。
n、k 是本机在 GF(2) 上重算的，与 CSV 的 n_data、k 对照。plpr、CI、shots 直接转抄 CSV，本报告不重测。

## 约定（与代码实现一致）

- **bb**：环 F2[x,y]/(x^ℓ−1, y^m−1)，x = S_ℓ ⊗ I_m，y = I_ℓ ⊗ S_m（S 为循环移位）。spec 里的项 [i,j] = x^i y^j。
  HX = [A | B]，HZ = [B^T | A^T]，n = 2ℓm。校验器把 ℓ ≤ m 规范化（必要时交换两轴）。
  ℓ = 1 时 x 平凡：该码是 C_m 上的单变量 generalized bicycle（GB）码。
- **hgp**：经典种子 H1（r1×n1）、H2（r2×n2）的 Tillich–Zémor 积：HX = [H1 ⊗ I_n2 | I_r1 ⊗ H2^T]，
  HZ = [I_n1 ⊗ H2 | H1^T ⊗ I_r2]，n = n1·n2 + r1·r2。两个种子都是环码（循环重复码）时，结果就是 n1×n2 环面上的 toric 码。
- **lp_abelian / lp_nonabelian**：基矩阵 A（mA×nA）、B（mB×nB）取值在 F2[G]，每个格子是一组群元素，群元素写成对
  catalog 顺序生成元的指数元组（阿贝尔群：每个因子一个指数，变量记作 x, y, …）。
  HX = [L(A ⊗ I_nB) | R(I_mA ⊗ B*)]，HZ = [R(I_nA ⊗ B) | L(A* ⊗ I_mB)]，L/R 是左/右正则表示，
  * = 转置 + 反极（g → g^-1）；n = (nA·nB + mA·mB)·|G|。阿贝尔群用因子表给出，非阿贝尔群用 catalog 名：
  D{n}：|G| = 2n，元组 [i,j] = r^i s^j；C{m}:C{n}(k)：s^-1 r s = r^k，元组 [i,j] = r^i s^j；
  C{m}xD{n}：元组 [a,i,j] = c^a r^i s^j。语法要求 A、B 支撑集的并生成整个 G。
  1×1 的 lifted product 就是两块群代数（2BGA）码，n = 2|G|。
- d：有 MILP 认证时给精确值（写作 d=），只认证一侧时给上界（d≤），否则是搜索阶段的代理值（d~）。

## 与文献写法的对照

- **BB**：Bravyi et al.（arXiv:2308.07915）表 3 的列是 [[n,k,d]]、ℓ,m、A、B，多项式直接写成 x^3 + y + y^2，
  并注明 HX = [A|B]，HZ = [B^T|A^T]，x^ℓ = y^m = 1，xy = yx。本报告的 bb 条目与之逐项对应。
- **2BGA / 非阿贝尔 1×1 LP**：Lin–Pryadko（arXiv:2306.16400）表 I 的列是 |G|、GAP 编号、n、k、d、a、b（按群表现写成生成元的字，
  如 1 + s r^4）、群表现 ⟨r,s | …⟩、结构描述；机器格式（github QEC-pages/2BGA-codes）用 GAP Elements(G) 的元素下标列表。
- **Mitten / 非阿贝尔 LP**：arXiv:2607.28795 表 I 给 [[n,k,d]] 与群名，表 VII 给群结构、GAP SmallGroup 编号、生成元关系与扭曲
  φ_b(a) = b a b^-1，附录 K 表 XIII 给基矩阵 A = [a0 ‖ a1]、B = [b0 ‖ b1]，每个元素用 GAP Elements(G) 的下标集合表示；
  阿贝尔 2×4 的码则写成单生成元 x 的多项式矩阵。
- **约定差异（重要）**：Mitten Def. 8 取 R(g): h ↦ h g^-1，本项目取 R(g): h ↦ h g。两者的 LP(A,B) 公式形式相同
  （HX = [L(A⊗I) | R(I⊗B*)]，HZ = [R(I⊗B) | L(A*⊗I)]），但因 R 的定义相差一个取逆，本项目的 LP(A, B) 等于 Mitten 约定下的
  LP(A, B̄)，B̄ 是 B **逐元素取逆、不转置**（已数值验证，逐字节相同）。论文若引用 Mitten 的定义，LP 条目应把 B 换成下面给出的 B̄，
  或在文中声明采用 h ↦ h g 的右正则表示。
- 本报告用 pc 正规形指数元组标识群元素（设计文档 §4.3），与 GAP Elements(G) 的下标之间需要 GAP 才能换算；GAP 编号已给出，
  读者可据表现关系在 GAP 中重建。
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_dir", type=Path)
    ap.add_argument("--budgets", nargs="+", default=["60", "98"])
    ap.add_argument("--site", default="v36")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--exact-d-json", type=Path, default=None,
                    help="certify_css_code_distance 的结果 {<arm>_b<budget>_r<rank>: {d_exact, certified, dX, dZ}}；有则把 d~ 换成精确 d")
    args = ap.parse_args(argv)
    exact_d = json.loads(args.exact_d_json.read_text()) if args.exact_d_json else {}
    out_dir = args.out or args.csv_dir
    tag = args.tag or "b" + "_b".join(args.budgets)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv_dir, set(args.budgets), args.site, args.top)
    if not rows:
        print("没有匹配的行", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    md: list[str] = []
    md.append(f"# 预算 {', '.join(args.budgets)}（site {args.site}）各臂 top-{args.top} 码的代数形式与 family 复核\n")
    md.append("下面每个码都按其 `spec` 用 campaign 自己的 family 构造器（`omniqec_agentic.construction`）重建过，"
              "重建的 CSS 矩阵对的哈希必须等于 CSV 的 `code_hash8`（所以写出的代数形式就是被测的那个码），"
              "n、k 在 GF(2) 上重算，并点名结构类别，便于凭肉眼判断 family 归属。\n")
    md.append(_CONVENTIONS)

    current = None
    for row in rows:
        v = verify(row)
        d = v["describe"]
        head = (row["arm"], row["budget"])
        if head != current:
            current = head
            done = "已跑完" if row["complete"] == "yes" else "未跑完（中间结果，会变）"
            md.append(f"\n## {row['arm']}  b{row['budget']}（模型 {row['model']}，{done}）\n")
        rank = int(row["rank"])
        title = "BB 基线（rank 0）" if rank == 0 else f"第 {rank} 名"
        flags = []
        if not v["hash_match"]:
            flags.append("哈希不一致")
        if not v["n_match"] or not v["k_match"]:
            flags.append("n/k 不一致")
        status = "已验证" if not flags else "、".join(flags)
        ed = exact_d.get(f"{row['arm']}_b{row['budget']}_r{rank}") or {}
        if ed.get("certified") and ed.get("d_exact") is not None:
            d_label, d_note = f"d={ed['d_exact']}", f"精确距离 d = {ed['d_exact']}（MILP 认证，d_X = {ed.get('dX')}, d_Z = {ed.get('dZ')}；搜索代理值 {row['d_search_proxy']}）"
        elif ed.get("d_exact") is not None:
            d_label, d_note = f"d≤{ed['d_exact']}", f"距离上界 d ≤ {ed['d_exact']}（只认证了一侧，d_X = {ed.get('dX')}, d_Z = {ed.get('dZ')}；代理值 {row['d_search_proxy']}）"
        else:
            d_label, d_note = f"d~{row['d_search_proxy']}", f"d 为搜索代理值 {row['d_search_proxy']}，未做精确认证"
        md.append(f"### {title}：`{row['program_id']}`  [[{v['n']},{v['k']},{d_label}]]  "
                  f"family={row['family']}  plpr={float(row['plpr']):.3e}  — {status}")
        md.append(f"- 距离：{d_note}")
        md.append(f"- 校验权重：w_X = {v['w_x']}，w_Z = {v['w_z']}；每个数据比特参与 {v['qubit_degree']} 个校验")
        md.append(f"- 结构类别：**{d['structural_class']}**")
        md.append(f"- 校验矩阵：{d['checks']}")
        if "weights" in d:
            md.append(f"- 重量：{d['weights']}")
        if "n" in d:
            md.append(f"- n：{d['n']}")
        if "group" in d:
            md.append(f"- 群 {d['group']}，|G| = {d['group_order']}，{d['group_identity']}，阿贝尔 = {d['group_is_abelian']}，"
                      f"A、B 支撑集生成 G = {d['supports_generate_G']}，基矩阵 A {d['A_shape']}、B {d['B_shape']}")
            if d.get("mitten_presentation"):
                md.append(f"- 群的表现（Mitten 表 VII 写法）：{d['mitten_presentation']}")
        if "seed_class" in d:
            md.append(f"- 种子：H1 = {d['seed_class']['H1']}，H2 = {d['seed_class']['H2']}")
        md.append(f"- 重建：n = {v['n']}（csv {row['n_data']}），k = {v['k']}（csv {row['k']}），"
                  f"CSS 正交 = {v['css_orthogonal']}，hash8 = {v['hash8_rebuilt']}（csv {row['code_hash8']}）")
        md.append(f"- 第 {row['turn']} 轮，shots {row['shots']}，CI [{float(row['ci_low']):.2e}, {float(row['ci_high']):.2e}]，vs_bb {row['vs_bb'] or '（无基线）'}")
        md.append("\n```\n    " + d["algebra"] + "\n```")
        md.append(f"spec：`{row['spec']}`\n")

        fname = f"{row['arm']}_b{row['budget']}_r{rank}_{row['code_hash8']}"
        np.savez_compressed(out_dir / "matrices" / f"{fname}.npz", hx=v["hx"], hz=v["hz"])
        records.append({
            "arm": row["arm"], "budget": int(row["budget"]), "site": row["site"], "model": row["model"],
            "complete": row["complete"], "rank": rank, "program_id": row["program_id"], "turn": row["turn"],
            "family": row["family"], "n": v["n"], "k": v["k"], "d_search_proxy": row["d_search_proxy"],
            "plpr": float(row["plpr"]), "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
            "code_hash8": row["code_hash8"], "hash_match": v["hash_match"], "n_match": v["n_match"],
            "k_match": v["k_match"], "css_orthogonal": v["css_orthogonal"],
            "w_x": v["w_x"], "w_z": v["w_z"], "qubit_degree": v["qubit_degree"],
            "exact_distance": exact_d.get(f"{row['arm']}_b{row['budget']}_r{rank}"),
            "spec": json.loads(row["spec"]), "validated_spec": v["validated_spec"],
            "algebra": {kk: vv for kk, vv in d.items() if kk != "algebra"}, "algebra_text": d["algebra"],
            "matrices_npz": f"matrices/{fname}.npz",
        })
        print(f"{row['arm']:<10} b{row['budget']:<4} r{rank}  {row['family']:<14} [[{v['n']},{v['k']}]]  "
              f"{status:<8} {d['structural_class'][:60]}")

    (out_dir / f"top_codes_{tag}.md").write_text("\n".join(md) + "\n")
    (out_dir / f"top_codes_{tag}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    n_bad = sum(1 for r in records if not (r["hash_match"] and r["n_match"] and r["k_match"]))
    print(f"\n{len(records)} 个码 -> {out_dir / f'top_codes_{tag}.md'}（{n_bad} 个不一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

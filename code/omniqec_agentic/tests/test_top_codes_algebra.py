"""hpc/top_codes_algebra: the structural-class naming and the rebuild-and-hash
proof that a merged CSV row's `spec` IS the measured code."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from omniqec_agentic.construction.build import build_matrices, compiled_code_hash
from omniqec_agentic.construction.registry import get_family
from omniqec_agentic.hpc import top_codes_algebra as tca

TORIC = {"family": "hgp", "seeds": {"H1": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
                                     "H2": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]}}
GB_AS_BB = {"A": [[0, 0], [0, 4], [0, 8]], "B": [[0, 0], [0, 5], [0, 13]], "ell": 1, "family": "bb", "m": 15}
BB = {"A": [[1, 0], [0, 1], [2, 4]], "B": [[0, 2], [1, 5], [2, 7]], "ell": 3, "family": "bb", "m": 8}
TBGA = {"A": [[[[0, 0, 0], [1, 2, 1]]]], "B": [[[[0, 0, 0], [1, 1, 0]]]], "family": "lp_nonabelian", "group": "C3xD4"}
LP = {"A": [[[[0], [1]]]], "B": [[[[0], [2]], [[0], [5]]], [[[0], [7]], [[0], [3]]]],
      "family": "lp_abelian", "group": {"factors": [12]}}


def _validated(spec):
    return get_family(spec["family"]).validate(spec)


def test_gf2_rank_matches_numpy_on_small_full_rank_and_defective_matrices() -> None:
    assert tca.gf2_rank(np.eye(4, dtype=np.uint8)) == 4
    assert tca.gf2_rank(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)) == 2  # ring code: rank n-1


def test_structural_classes_are_named_beyond_the_family_label() -> None:
    assert tca.describe(_validated(TORIC))["structural_class"].startswith("toric 码")
    assert "3×4 环面" in tca.describe(_validated(TORIC))["structural_class"]
    assert tca.describe(_validated(GB_AS_BB))["structural_class"].startswith("单变量")
    assert tca.describe(_validated(BB))["structural_class"] == "C3×C8 上的双变量 bicycle（BB）码"
    d = tca.describe(_validated(TBGA))
    assert d["structural_class"].startswith("1×1 lifted product")
    assert d["group_is_abelian"] is False and d["supports_generate_G"] is True
    assert "c r^2 s" in d["algebra"]  # tuple [1,2,1] over C3xD4 = c r^2 s
    assert tca.describe(_validated(LP))["structural_class"] == "群 C12 上的 lifted product，基矩阵 A 为 1×1、B 为 2×2"
    assert "(1 + x^7)  (1 + x^3)" in tca.describe(_validated(LP))["algebra"]


def test_bb_polynomials_render_with_the_x_y_convention() -> None:
    d = tca.describe(_validated(BB))
    assert d["A"] == "x + y + x^2 y^4" and d["B"] == "y^2 + x y^5 + x^2 y^7"


def _row(spec, arm="testarm", budget="60", rank="1", **over):
    hx, hz = build_matrices(spec)
    n = int(hx.shape[1])
    k = n - tca.gf2_rank(hx) - tca.gf2_rank(hz)
    row = {
        "arm": arm, "budget": budget, "site": "v36", "model": "m", "complete": "yes", "rank": rank,
        "turn": "3", "program_id": "g0001_p0001", "family": spec["family"], "n_data": str(n),
        "n_checks_x": str(hx.shape[0]), "n_checks_z": str(hz.shape[0]), "k": str(k), "d_search_proxy": "4",
        "selection_channels": "", "plpr": "0.001", "ci_low": "0.0009", "ci_high": "0.0011", "vs_bb": "",
        "block_ler": "", "failures": "1", "shots": "1000", "p": "0.005", "rounds": "4", "decoder": "t",
        "code_hash8": compiled_code_hash(hx, hz)[:8], "spec": json.dumps(spec, separators=(",", ":")),
    }
    row.update(over)
    return row


def test_end_to_end_rebuilds_hashes_and_writes_report(tmp_path: Path) -> None:
    rows = [_row(TORIC, rank="1"), _row(BB, rank="2"), _row(TBGA, budget="98", rank="1"),
            _row(LP, budget="98", rank="2", code_hash8="deadbeef")]  # last one: wrong hash on purpose
    with open(tmp_path / "fig2_all_best_m.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert tca.main([str(tmp_path), "--budgets", "60", "98", "--site", "v36"]) == 0
    records = json.loads((tmp_path / "top_codes_b60_b98.json").read_text())
    assert [r["hash_match"] for r in records] == [True, True, True, False]
    assert all(r["n_match"] and r["k_match"] and r["css_orthogonal"] for r in records)
    md = (tmp_path / "top_codes_b60_b98.md").read_text()
    assert "toric 码" in md and "哈希不一致" in md and md.count("— 已验证") == 3
    npz = np.load(tmp_path / records[0]["matrices_npz"])
    assert npz["hx"].shape[1] == records[0]["n"] == 24

"""Static guards for the v3.5 decisions recorded in the architecture and SEC-design documents.

The v3.5 launcher / submitter / status trio and its runbook were historical
deployment files and are not part of the open-source tree. The two decisions
they carried are still documented and pinned here: the noise protocol of the
OmniQEC paper SI C (noise profile uniform_paper_v1 at the single p=0.005, NO
idle noise) and the SEC policy twobasis_nonint_v1 (both bases measured every
round, no interleaving, K=16 intra-basis layer orders certified with 400 ISD
iterations each), a CODE DEFAULT rather than a launcher flag."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = PACKAGE_ROOT / "docs" / "04_architecture.md"
SEC_DESIGN = PACKAGE_ROOT / "docs" / "02_sec_design.md"

# OmniQEC paper SI C b "Noise model", quoted verbatim (single line) in every
# v3.5 document; the lowering in sec/stim_builder.py is derived from it.
SI_C_SENTENCE = (
    "All circuits use a uniform circuit-level depolarizing model controlled by a "
    "single physical error rate p. Each qubit reset is followed by single-qubit "
    "depolarizing noise of strength p; each controlled-NOT is followed by two-qubit "
    "depolarizing noise of strength p; each ancilla measurement is preceded by a "
    "bit-flip error of probability p; and the final data readout is preceded by a "
    "basis-conjugate flip of probability p."
)


def test_v35_docs_quote_the_paper_noise_model_verbatim() -> None:
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    sec_design = SEC_DESIGN.read_text(encoding="utf-8")
    # The 2026-08-31 repair plan quoted it too, but it was an account-local file that never
    # entered the repository; the two tracked documents are checked.
    for name, text in (("04", architecture), ("02", sec_design)):
        assert SI_C_SENTENCE in text, name
        assert "uniform_paper_v1" in text, name
        assert "bposd_structural_v3_5" in text, name
        # The exact lowering, stated in each document.
        assert "DEPOLARIZE1" in text and "DEPOLARIZE2" in text, name
    # 04: header profile, the new section, the launcher row and the two R entries.
    header_line = architecture.splitlines()[2]
    assert "bposd_structural_v3_5" in header_line and "追加（六）" in header_line
    assert "2026-09-02 追加（六）" in architecture
    assert "nscc_claude_grid_v35.pbs" in architecture
    assert "submit_claude_v35_grid.sh" in architecture
    assert "claude_v35_status.sh" in architecture
    r17 = architecture[architecture.index("| R17 |") : architecture.index("\n", architecture.index("| R17 |"))]
    assert "nscc_claude_grid_v35.pbs" in r17
    assert "| R21 |" in architecture
    assert "不可合池" in architecture
    # 02: the addendum keeps the idle rule for uniform_legacy_v1 history/golden only.
    assert "### 9.3" in sec_design
    assert "uniform_legacy_v1" in sec_design[sec_design.index("### 9.3") :]


def test_v35_docs_record_the_twobasis_sec_policy() -> None:
    """The SEC switch (same day, second decision) is documented in 04 as a new
    appendix section with the evidence table, the exact timeline / detector /
    certification rules and the non-claims, in 02 as addendum 9.4, and in the
    header line of 04."""

    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    sec_design = SEC_DESIGN.read_text(encoding="utf-8")
    header_line = architecture.splitlines()[2]
    assert "twobasis_nonint_v1" in header_line and "追加（七）" in header_line
    section = architecture[architecture.index("### 2026-09-02 追加（七）") : architecture.index("## 11. 已知残留")]
    # Evidence table (2026-09-02, p=0.5%, no idle) and the user's decisions.
    assert "0.98 / 0.97" in section and "0.002 / 0.004" in section
    assert "0 / 1040 shots" in section
    assert "不做任何的交错" in section
    # Timeline / detector rule / certification constants, verbatim names.
    assert "sec/twobasis_builder.py::build_twobasis_memory_circuit" in section
    assert "|LX|+|LZ|+4" in section
    assert "T·m_W + (T−1)·m_opp + m_W" in section
    assert "sec/layers.py" in section and "color_bipartite_multigraph" in section
    assert "sec/twobasis_rank.py::rank_twobasis_schedules" in section
    for token in ("sec_policy_id=\"twobasis_nonint_v1\"", "sec_candidates=16", "sec_cert_iters=400", "sec_cert_seed=0"):
        assert token in section, token
    assert "SBB Thm 1" in section
    # The two non-claims the contract insists on.
    assert "ISD 估计是上界" in section
    assert "hook 错误**没有被消除**" in section
    assert "不可合池" in section
    # Freeze / validator / launcher rows and the residual entry.
    assert "FORMAL_SEC_POLICY" in section and "V35_DEFAULTS_OK" in section
    assert "| R22 |" in architecture
    assert "### 9.4" in sec_design
    addendum = sec_design[sec_design.index("### 9.4") :]
    assert "twobasis_nonint_v1" in addendum and "lrc_ranked_v1" in addendum
    assert "ISD 估计是上界" in addendum

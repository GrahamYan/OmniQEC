"""Static guards for the Claude v3.6 NSCC deployment scripts.

The v3.5 trio and the deployment runbooks / site README were historical
files and are not part of the open-source tree.

This file pins the v3.6 launcher / submitter / status scripts and the
requirements files to the telescoping_structural_v3_7 contract (docs/06,
user decision 2026-09-03): ONE decoder telescoping_cpu --
the Mitten paper's open-source telescoping decoder (vendored at
third_party/telescoping-decoder) in its CPU configuration: S1/S2 off, S3-A/B/C
on the GARI system, S4 off, an undecoded shot (NC) counts as a logical error,
no BPOSD -- the budget policy applied from --budget (shots only, no per-stage
time limits; no --slow-shots / --final-shots / FINAL_SHOTS_OVERRIDE in the
launcher), the S3 kernels PREBUILT once in the prologue into a job-visible
XDG_CACHE_HOME that every worker inherits, OMP_NUM_THREADS=1 and the memory
cap env in every worker, the v3.4 execution geometry unchanged, every
FATAL_STOP branch unchanged, the noise pair passed explicitly, and the
V37_DEFAULTS_OK startup self-check (incl. the vendored commit).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
HPC = PACKAGE_ROOT / "hpc"
PBS = HPC / "nscc_claude_grid_v36.pbs"
STATUS = HPC / "claude_v36_status.sh"
SUBMIT = HPC / "submit_claude_v36_grid.sh"
# The GPT arm (2026-09-04): the same launcher with ARM=gpt, ONE serial chain.
SUBMIT_GPT = HPC / "submit_gpt_v36_serial.sh"
# The DeepSeek arm on NSCC (2026-09-10): the same launcher with ARM=deepseek
# (OpenRouter, pay per token), the Claude submitter with ARM=deepseek, and a
# status wrapper.
STATUS_DEEPSEEK = HPC / "deepseek_v36_status.sh"
STATUS_GPT = HPC / "gpt_v36_status.sh"
NSCC_REQUIREMENTS = HPC / "requirements_fig2_nscc.txt"
PACKAGE_REQUIREMENTS = PACKAGE_ROOT / "requirements.txt"
VENDORED = REPO_ROOT / "third_party" / "telescoping-decoder"

EDITABLE_INSTALL = "pip install -e ./third_party/telescoping-decoder"
KERNEL_PREBUILD = "python -m telescoping_decoder.build"


def _call_block(text: str) -> str:
    campaign_call = text.index("python -m omniqec_agentic.cli run-campaign")
    return text[campaign_call : text.index("9>&-", campaign_call)]


def _self_check_block(text: str) -> str:
    self_check = text.index("print('V37_DEFAULTS_OK')")
    return text[text.rindex("if ! python -c", 0, self_check) : self_check + 400]


def test_v36_scripts_exist() -> None:
    for path in (PBS, STATUS, SUBMIT, SUBMIT_GPT, STATUS_GPT, STATUS_DEEPSEEK):
        assert path.is_file(), f"missing v3.6 script: {path}"


def test_v36_launcher_freezes_geometry_and_passes_no_shots() -> None:
    text = PBS.read_text(encoding="utf-8")
    assert "#PBS -N omni_claude_v36" in text
    assert "#PBS -N omni_claude_v35" not in text
    assert "#PBS -l select=1:ncpus=128:mem=100gb" in text
    assert 'RUN_TAG="${RUN_TAG:-v37}"' in text
    # 2026-09-04: 64- or 128-core nodes; workers = cores - 1, and each build
    # process gets a quarter of them for its SEC certification probes.
    assert "readonly WORKERS=$((ALLOCATED_CPUS - 1))" in text
    assert "readonly WORKERS=127" not in text
    # 2026-10-16: 127 (NSCC q3, one core idle) joined the whole-node sizes.
    assert "(( ALLOCATED_CPUS != 128 )) && (( ALLOCATED_CPUS != 127 )) && (( ALLOCATED_CPUS != 64 ))" in text
    assert 'export OMNIQEC_SEC_WORKERS="${OMNIQEC_SEC_WORKERS:-$(( WORKERS / 4 ))}"' in text
    assert "sec_workers_per_build=$OMNIQEC_SEC_WORKERS" in text
    # 2026-09-05: slow_workers is a runtime key; a different core count is a note, not exit 2
    assert "resubmit this point with NCPUS=" not in text
    assert "runtime override, recorded in protocol_snapshot.json" in text
    assert 'export OMNIQEC_SHARD_SPLIT="${OMNIQEC_SHARD_SPLIT:-$(( ALLOCATED_CPUS >= 128 ? 8 : 4 ))}"' in text
    assert "shard_split=${OMNIQEC_SHARD_SPLIT:-auto}" in text
    assert 'REPO_ROOT="${REPO_ROOT:-$HOME/OmniQEC}"' in text
    assert "readonly SHOT_SHARDS=16" in text
    assert "readonly FINAL_SHOT_SHARDS=128" in text
    assert "readonly DECODE_SHOT_BUDGET_S=8" in text
    assert "readonly BUILD_TIMEOUT_S=3600" in text
    assert "readonly DECODE_SHARD_TIMEOUT_S=1800" in text
    assert "readonly DISTANCE_TIMEOUT_S=1800" in text
    assert "readonly TIMEOUT_RETRIES=0" in text
    assert "readonly TRANSIENT_RETRIES=1" in text
    assert 'DECODERS="telescoping_cpu"' in text
    assert 'DECODERS="relay_cascade_gari"' not in text
    assert 'DECODERS="bposd"' not in text
    call_block = _call_block(text)
    for flag in (
        '--shot-shards "$SHOT_SHARDS"',
        '--final-shot-shards "$FINAL_SHOT_SHARDS"',
        '--decode-shot-budget-s "$DECODE_SHOT_BUDGET_S"',
        '--transient-retries "$TRANSIENT_RETRIES"',
        '--build-timeout-s "$BUILD_TIMEOUT_S"',
        '--decode-shard-timeout-s "$DECODE_SHARD_TIMEOUT_S"',
        '--distance-timeout-s "$DISTANCE_TIMEOUT_S"',
        '--timeout-retries "$TIMEOUT_RETRIES"',
        "--decoders $DECODERS",
        "--strict-turns",
        '--noise-profile "$NOISE_PROFILE"',
        '--p "$P_VALUE"',
    ):
        assert flag in call_block, flag
    # v3.6: shots derive from --budget inside run-campaign (budget policy).
    assert "--final-shots" not in call_block
    assert "--slow-shots" not in call_block
    assert "--racing" not in call_block
    assert "--sec-policy" not in call_block
    assert "relay_cascade_gari" not in call_block and "cascade" not in call_block
    assert "--circuit-timeout-s" not in text
    assert "shots_for()" not in text
    assert "policy_shots()" in text
    assert "from omniqec_agentic.config import budget_policy" in text
    # The policy helper prints shots only: there are no S4 time limits.
    assert "s4_time" not in text
    assert "S4_SEARCH_S" not in text and "S4_FINAL_S" not in text
    assert 'read -r SEARCH_SHOTS FINAL_SHOTS < <(policy_shots "$BUDGET")' in text
    assert "telescoping_structural_v3_7" in text
    assert '"$RUN_TAG" != "v37"' in text
    assert '"$RUN_TAG" != "v35"' not in text
    # The run-dir prefix is the arm's; ARM defaults to claude, so the Claude
    # arm's paths are unchanged (fig2_claudesub_b<N>_v37).
    assert 'OUT_DIR="$OUT_ROOT/fig2_${ARM_PREFIX}_b${BUDGET}_${RUN_TAG}"' in text
    assert 'ARM="${ARM:-claude}"' in text
    assert 'claude) ARM_PREFIX="claudesub"; ARM_LABEL="Claude";   LLM_MODEL="$CLAUDE_MODEL"; PROVIDER="openai" ;;' in text
    assert "fig2_claudesub_b${BUDGET}" not in text
    # The banner names the protocol / decoder the way the snapshot freezes it.
    assert "protocol=telescoping_structural_v3_7" in text
    assert "protocol=cascade_structural_v3_6" not in text
    assert "decoder: telescoping_cpu = telescoping_cpu_v1" in text
    assert "timeouts (floor + model)" in text
    assert "shots x telescoping_cpu worst-case shot" in text


def test_v36_launcher_refuses_the_retired_final_shots_override() -> None:
    text = PBS.read_text(encoding="utf-8")
    guard = text.index('if [[ -n "${FINAL_SHOTS_OVERRIDE:-}" ]]; then')
    assert "is not a v3.6 knob" in text[guard : guard + 400]
    assert "exit 2" in text[guard : guard + 400]
    assert 'FINAL_SHOTS_OVERRIDE="${FINAL_SHOTS_OVERRIDE:-}"' not in text
    assert "FINAL_SHOTS_OVERRIDE < FINAL_SHOT_SHARDS" not in text
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "is not a v3.6 knob" in submit
    assert 'common_vars+=",FINAL_SHOTS_OVERRIDE' not in submit


def test_v36_launcher_sets_single_thread_workers_and_the_memory_cap() -> None:
    text = PBS.read_text(encoding="utf-8")
    campaign_call = text.index("python -m omniqec_agentic.cli run-campaign")
    for line in (
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        'export OMNIQEC_TOTAL_MEM_BYTES="${OMNIQEC_TOTAL_MEM_BYTES:-$((100 * 1024 * 1024 * 1024))}"',
        'export OMNIQEC_WORKER_RSS_BYTES="${OMNIQEC_WORKER_RSS_BYTES:-$((768 * 1024 * 1024))}"',
    ):
        assert line in text, line
        assert text.index(line) < campaign_call
    # No BPOSD-specific environment survives.
    assert "OSD" not in text.replace("osd_cs", "").replace("BPOSD", "").replace("bposd", "")
    assert "workers: OMP_NUM_THREADS=$OMP_NUM_THREADS" in text


def test_v36_launcher_prebuilds_the_telescoping_kernels_before_any_worker() -> None:
    """The package compiles its two C kernels on first use into
    $XDG_CACHE_HOME/telescoping_decoder. The launcher pins that cache to a
    shared-filesystem path (exported before run-campaign, so every worker and
    the package's S3 child processes inherit it), loads the gcc module, and
    runs the prebuild ONCE in the prologue -- after the self-check, before the
    first run-campaign -- failing the job by name when it does not build."""

    text = PBS.read_text(encoding="utf-8")
    campaign_call = text.index("python -m omniqec_agentic.cli run-campaign")
    self_check = text.index("print('V37_DEFAULTS_OK')")
    export = text.index('export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/runs/.cache}"')
    gcc = text.index("module load gcc")
    prebuild = text.index(f"if ! {KERNEL_PREBUILD}; then")
    assert text.index('REPO_ROOT="${REPO_ROOT:-$HOME/OmniQEC}"') < export < self_check < prebuild < campaign_call
    assert text.index("module load miniforge3/23.10") < gcc < export
    assert 'mkdir -p "$XDG_CACHE_HOME"' in text[export:prebuild]
    failure = text[prebuild : text.index("fi", prebuild)]
    assert "telescoping S3 kernel prebuild FAILED" in failure
    assert EDITABLE_INSTALL.split("pip install ", 1)[1] in failure  # names the remedy
    assert "module load gcc" in failure
    assert "exit 4" in failure
    assert "telescoping S3 kernels ready in $XDG_CACHE_HOME/telescoping_decoder" in text
    # The cache is never pointed at a node-local path by the launcher itself.
    assert 'XDG_CACHE_HOME="/tmp' not in text and "XDG_CACHE_HOME=/tmp" not in text
    assert text.count(KERNEL_PREBUILD) >= 1


def test_v36_startup_self_check_asserts_the_code_defaults() -> None:
    text = PBS.read_text(encoding="utf-8")
    freshness = text.index("from omniqec_agentic.hpc.validate_fig2_campaign import FORMAL_FINAL_SHOT_SHARDS")
    self_check = text.index("print('V37_DEFAULTS_OK')")
    campaign_call = text.index("python -m omniqec_agentic.cli run-campaign")
    assert freshness < self_check < campaign_call
    block = _self_check_block(text)
    for literal in (
        "NoiseSweepConfig().profile.value == 'uniform_paper_v1'",
        "StructuralSearchConfig().profile_id == 'telescoping_structural_v3_7'",
        "PROTOCOL_PROFILE == 'telescoping_structural_v3_7' == FORMAL_PROFILE",
        "StructuralSearchConfig().reducer_id == 'structural_proxy_pareto_v3_weight'",
        "v.decoders == (DecoderName.TELESCOPING,) and DecoderName.TELESCOPING.value == 'telescoping_cpu' == FORMAL_DECODER",
        "telescoping = TelescopingDecoderConfig().to_dict()",
        "assert telescoping == TELESCOPING_PROTOCOL_FIELDS, telescoping",
        "telescoping['preset'] == 'telescoping_cpu_v1' and telescoping['nc_policy'] == 'fail'",
        "telescoping['s3_system'] == 'gari' and not (telescoping['s1_enabled'] or telescoping['s2_enabled'] or telescoping['s4_enabled'])",
        "v.decode_shot_budget_s == 8.0 and v.timeout_retries == 0 and v.transient_retries == 1",
        "FORMAL_SHOT_SHARDS == 16 and FORMAL_FINAL_SHOT_SHARDS == 128",
        "budget_policy(288) == {'large_budget': False, 'search_shots': 25000, 'final_shots': 25000}",
        "budget_policy(360) == {'large_budget': False, 'search_shots': 25000, 'final_shots': 25000}",
        "budget_policy(400) == {'large_budget': True, 'search_shots': 5000, 'final_shots': 50000}",
        "import telescoping_decoder",
        "Path(telescoping_decoder.__file__).resolve().parent.parent / 'VENDORED_COMMIT'",
        "assert TELESCOPING_VENDORED_COMMIT in vendored.read_text(encoding='utf-8')",
        "DEFAULT_SEC_POLICY.sec_policy_id == 'twobasis_nonint_v1' == FORMAL_SEC_POLICY",
        "(DEFAULT_SEC_POLICY.sec_candidates, DEFAULT_SEC_POLICY.sec_cert_iters) == (16, 400)",
        "(DEFAULT_SEC_POLICY.sec_cert_seeds, DEFAULT_SEC_POLICY.sec_refine_top, DEFAULT_SEC_POLICY.sec_refine_iters) == (3, 4, 2000)",
        "assert_resumable_protocol({'protocol_profile': 'bposd_structural_v3_5', 'decoder': 'bposd'})",
        "exit 4",
    ):
        assert literal in block, literal
    for retired in ("CASCADE", "cascade", "relay_cascade_gari", "s4_time", "fallback"):
        assert retired not in block, retired
    assert "print('V35_DEFAULTS_OK')" not in text
    assert "bposd_structural_v3_5" in text[: text.index("#PBS -P")]  # named as history
    assert "telescoping_cpu" in text[self_check : self_check + 600]  # the failure message names the protocol


def test_v36_self_check_literals_match_the_code_constants() -> None:
    from omniqec_agentic.config import (
        DEFAULT_SEC_POLICY,
        TELESCOPING_PROTOCOL_FIELDS,
        TELESCOPING_VENDORED_COMMIT,
        DecoderName,
        StructuralSearchConfig,
        TelescopingDecoderConfig,
        VerificationConfig,
        budget_policy,
    )
    from omniqec_agentic.hpc.validate_fig2_campaign import FORMAL_DECODER, PROTOCOL_PROFILE
    from omniqec_agentic.workflow.state import FORMAL_PROFILE

    assert PROTOCOL_PROFILE == FORMAL_PROFILE == StructuralSearchConfig().profile_id == "telescoping_structural_v3_7"
    assert FORMAL_DECODER == DecoderName.TELESCOPING.value == "telescoping_cpu"
    assert VerificationConfig().decoders == (DecoderName.TELESCOPING,)
    telescoping = TelescopingDecoderConfig().to_dict()
    assert telescoping == TELESCOPING_PROTOCOL_FIELDS
    assert telescoping["preset"] == "telescoping_cpu_v1" and telescoping["nc_policy"] == "fail"
    assert telescoping["s3_system"] == "gari"
    assert not (telescoping["s1_enabled"] or telescoping["s2_enabled"] or telescoping["s4_enabled"])
    assert budget_policy(288) == {"large_budget": False, "search_shots": 25000, "final_shots": 25000}
    assert budget_policy(360) == {"large_budget": False, "search_shots": 25000, "final_shots": 25000}
    assert budget_policy(400) == {"large_budget": True, "search_shots": 5000, "final_shots": 50000}
    assert (DEFAULT_SEC_POLICY.sec_cert_seeds, DEFAULT_SEC_POLICY.sec_refine_top, DEFAULT_SEC_POLICY.sec_refine_iters) == (3, 4, 2000)
    # The vendored commit the self-check reads next to the package.
    vendored = VENDORED / "VENDORED_COMMIT"
    assert vendored.is_file(), vendored
    assert TELESCOPING_VENDORED_COMMIT in vendored.read_text(encoding="utf-8")
    assert (VENDORED / "telescoping_decoder" / "_c" / "build.py").is_file()


def test_v36_fatal_marker_and_all_infeasible_stop_are_unchanged() -> None:
    text = PBS.read_text(encoding="utf-8")
    marker_check = text.index('if [[ -e "$FATAL_MARKER" ]]')
    campaign_call = text.index("python -m omniqec_agentic.cli run-campaign")
    measurement_stop = text.index('write_fatal_stop "$OUT_DIR" "measurement_blocked"')
    all_infeasible_grep = text.index('grep -qF "no successful final measurement rows" "$ATTEMPT_LOG"')
    all_infeasible_stop = text.index('write_fatal_stop "$OUT_DIR" "final_all_infeasible"')
    strict_retry = text.index("if looks_like_strict_abort")
    assert marker_check < campaign_call < measurement_stop < all_infeasible_grep < all_infeasible_stop < strict_retry
    header = text[: text.index("#PBS -P")]
    for reason in (
        "measurement_blocked",
        "final_all_infeasible",
        "strict_retries_exhausted",
        "non_quota_campaign_failure",
        "formal_validation_failed",
    ):
        assert f'write_fatal_stop "$OUT_DIR" "{reason}"' in text, reason
        assert reason in header, reason
    assert 'write_fatal_stop "$OUT_DIR" "decode_infeasible"' not in text
    # A signal (walltime / qdel) ends the link instead of being filed as a
    # non-quota failure, and a signal-killed run-campaign never writes a marker.
    assert "trap on_signal INT TERM" in text and "trap cleanup EXIT\n" in text
    assert "trap cleanup EXIT INT TERM" not in text
    on_signal = text.index("on_signal() {")
    assert "exit 143" in text[on_signal : on_signal + 400]
    signal_guard = text.index("if (( rc >= 128 )); then")
    assert campaign_call < signal_guard < measurement_stop
    assert "no FATAL_STOP; the next chain link resumes from the checkpoint" in text[signal_guard : signal_guard + 400]
    assert "every FATAL_STOP branch" in header
    assert "telescoping_cpu" in header and "budget_policy" in header
    for phrase in ("S1/S2", "S3-A/B/C", "S4", "COUNTS AS A LOGICAL ERROR", "per-stage time limits"):
        assert phrase in header, phrase
    assert "S4 60 / 600 s" not in header and "S4 30 / 300 s" not in header


def test_v36_validator_call_states_the_decoder_and_not_the_shots() -> None:
    text = PBS.read_text(encoding="utf-8")
    start = text.index("python omniqec_agentic/hpc/validate_fig2_campaign.py")
    block = text[start : text.index("; then", start)]
    assert "--decoders $DECODERS" in block
    assert '--p "$P_VALUE"' in block
    assert "--final-shots" not in block
    assert '"completed campaign failed the standalone v3.6 artifact validator"' in text


def test_v36_submitter_uses_independent_afterany_chains_for_current_grid() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    assert 'POINTS="${POINTS:-360 288 260 240 220 200 180 160 140 120 98 80 60}"' in text
    assert 'CHAIN_LINKS="${CHAIN_LINKS:-4}"' in text
    assert 'RUN_TAG="${RUN_TAG:-v37}"' in text
    assert "nscc_claude_grid_v36.pbs" in text
    assert "nscc_claude_grid_v35.pbs" not in text
    assert "depend=afterany:$previous" in text
    assert '[[ "$job_name" == "$JOB_NAME" ]]' in text and 'JOB_NAME="omni_${ARM}_v36"' in text
    assert "omni_claude_v35" not in text
    assert "refusing duplicate $ARM_LABEL v3.6 submission" in text
    assert 'vars="POINTS=$point,$common_vars"' in text
    assert '"$RUN_TAG" != "v37"' in text
    assert "OMNIQEC_WORKER_RSS_BYTES" in text and "OMNIQEC_TOTAL_MEM_BYTES" in text
    # NCPUS=64|128 per submission, frozen-worker pre-check, SEC workers passthrough.
    assert 'NCPUS="${NCPUS:-128}"' in text
    assert '[[ "$NCPUS" == "128" || "$NCPUS" == "64" ]]' in text
    assert 'qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -v "$vars" "$PBS_SCRIPT"' in text
    # -P overrides the launcher's #PBS -P placeholder; the project has no default (open-source tree)
    assert 'PROJ="${PROJ:-}"' in text
    assert 'set PROJ=<PROJECT> (the PBS project to charge)' in text
    assert "#PBS -P <PROJECT>" in PBS.read_text(encoding="utf-8")
    # 2026-09-05: a different core count is a note (slow_workers is a runtime key), never a refusal
    assert "refusing: NCPUS=$NCPUS does not match" not in text
    assert "note: NCPUS=$NCPUS differs from the worker count these points last ran with" in text
    assert "REPO_ROOT=$REPO_ROOT" in text and "OMNIQEC_SHARD_SPLIT" in text
    assert "OMNIQEC_SEC_WORKERS" in text
    header = text[: text.index("set -euo pipefail")]
    assert "POINTS=144 TURNS=2 RUN_TAG=v37_canary" in header
    assert "VALIDATE_FORMAL=0 CHAIN_LINKS=2" in header
    assert "must NOT be rsynced" in header
    assert "telescoping_cpu" in header and "budget_policy" in header
    assert "relay_cascade_gari" not in text and "S5" not in text and "S4 60/600" not in text


def test_v36_status_filters_the_v36_job_and_reports_decoder_policy_and_rss() -> None:
    text = STATUS.read_text(encoding="utf-8")
    assert 'RUN_TAG="${RUN_TAG:-v37}"' in text
    # Arm-parameterised: ARM defaults to claude, so the job filter is still
    # omni_claude_v36 and the run-dir glob still fig2_claudesub_b*_v36.
    assert 'ARM="${ARM:-claude}"' in text
    assert 'JOB_NAME="omni_${ARM}_v36"' in text
    assert '[[ "$job_name" == "$JOB_NAME" ]]' in text
    assert 'claude) ARM_PREFIX="claudesub"; ARM_LABEL="Claude" ;;' in text
    assert 'gpt)    ARM_PREFIX="gptsub";    ARM_LABEL="ChatGPT" ;;' in text
    assert 'deepseek) ARM_PREFIX="deepseek"; ARM_LABEL="DeepSeek (OpenRouter)" ;;' in text
    assert 'for d in "$REPO"/runs/fig2_"${ARM_PREFIX}"_b*_"$RUN_TAG"; do' in text
    assert 'LOG_PREFIX="${ARM}_grid_${RUN_TAG}"' in text
    assert "omni_claude_v35" not in text
    assert "${ARM_LABEL} v3.6 grid" in text
    assert "search 16 / final 128" in text
    assert '[[ "$shards" == "16" ]]' in text
    assert '[[ "$final_shards" == "128" ]]' in text
    assert '[[ "$noise_profile" == "uniform_paper_v1" ]]' in text
    assert '[[ "$decoder" == "telescoping_cpu" ]]' in text
    assert '[[ "$preset" == "telescoping_cpu_v1" ]]' in text
    assert "relay_cascade_gari" not in text
    assert "⚠decoder=" in text and "⚠noise=" in text and "⚠telescoping=" in text
    assert 'p.get("budget_policy")' in text and 'p.get("decoder")' in text and 'p.get("telescoping")' in text
    assert "telescoping=${preset}" in text
    assert "shard_max_rss_bytes" in text and "peak_rss" in text
    assert "decode_infeasible" in text and "slow_summary.json" in text
    assert "FATAL_STOP.json" in text
    # The Chinese explanation names the protocol decoder and the NC rule.
    assert "正式值是 telescoping_cpu" in text
    assert "NC" in text and "逻辑错误" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_v36_scripts_parse_under_bash_n() -> None:
    for path in (PBS, SUBMIT, STATUS, SUBMIT_GPT, STATUS_GPT, STATUS_DEEPSEEK):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{path}: {proc.stderr}"


def test_v36_requirements_document_the_vendored_decoder() -> None:
    """The formal decoder is not on PyPI: both requirements files point at the
    editable install of the vendored package, name the C compiler requirement
    and the kernel prebuild, and demote relay-bp to the dev-only relaybp
    decoder."""

    nscc = NSCC_REQUIREMENTS.read_text(encoding="utf-8")
    for token in (
        "telescoping_structural_v3_7",
        "telescoping_cpu",
        "third_party/telescoping-decoder",
        "python -m pip install -e ./third_party/telescoping-decoder",
        KERNEL_PREBUILD,
        "module load gcc",
        "XDG_CACHE_HOME",
        "OPTIONAL",
    ):
        assert token in nscc, token
    assert "relay_cascade_gari" not in nscc and "HiGHS" not in nscc
    package = PACKAGE_REQUIREMENTS.read_text(encoding="utf-8")
    assert EDITABLE_INSTALL in package
    assert KERNEL_PREBUILD in package
    assert "OPTIONAL" in package.split("relay-bp", 1)[1].splitlines()[0]
    assert "telescoping_structural_v3_7" in package


# --- GPT arm: the same launcher, ARM=gpt, one serial chain (2026-09-04) ----

def test_v36_launcher_runs_the_gpt_arm_through_the_same_protocol() -> None:
    """ARM=gpt changes the model, the run-dir/log prefixes and the endpoint
    lookup -- nothing about the protocol. The gateway URL may carry a path
    token and is never echoed; a guessed model id fails by name at startup."""

    text = PBS.read_text(encoding="utf-8")
    for token in (
        'GPT_MODEL="${GPT_MODEL:-gpt-5.6-sol}"',
        'gpt)    ARM_PREFIX="gptsub";    ARM_LABEL="ChatGPT";  LLM_MODEL="$GPT_MODEL"; PROVIDER="openai" ;;',
        '*) echo "ARM must be claude, gpt or deepseek (got ARM=$ARM)" >&2; exit 2 ;;',
        'POINTS="${POINTS//:/ }"',
        'FATAL_POLICY="${FATAL_POLICY:-stop}"',
        'QUOTA_MAX_WAIT_S="${QUOTA_MAX_WAIT_S:-0}"',
        'GPT_ENDPOINT_FILE="${GPT_ENDPOINT_FILE:-$HOME/.cli-proxy-api/gpt_endpoint.env}"',
        '[[ "$ARM" == "gpt" && -f "$GPT_ENDPOINT_FILE" ]] && ARM_ENDPOINT_FILE="$GPT_ENDPOINT_FILE"',
        '[[ "$ARM" == "claude" && -f "$CLAUDE_ENDPOINT_FILE" ]] && ARM_ENDPOINT_FILE="$CLAUDE_ENDPOINT_FILE"',
        'CLAUDE_ENDPOINT_FILE="${CLAUDE_ENDPOINT_FILE:-$HOME/.cli-proxy-api/claude_endpoint.env}"',
        '[[ "$OPENAI_BASE_URL" == */v1 ]]',
        'if (( ! GATEWAY_MODE && ! OPENROUTER_MODE )) && [[ -f "$SHARED_ENDPOINT_FILE" ]]; then',
        'if (( ! SHARED_MODE && ! GATEWAY_MODE && ! OPENROUTER_MODE )); then',
        'exec > >(tee -a "$LOG_ROOT/${ARM}_grid_${RUN_TAG}.${JOB_TAG}.log") 2>&1',
        'ATTEMPT_LOG="$LOG_ROOT/${ARM_PREFIX}_b${BUDGET}_${RUN_TAG}.${JOB_TAG}.attempt.log"',
        'export OPENAI_MODEL="$LLM_MODEL"',
        '--provider "$PROVIDER" --model "$LLM_MODEL"',
        'fatal_link_exit() { [[ "$FATAL_POLICY" == "skip" ]] || exit 1; }',
        "is not in this endpoint's /v1/models",
        "model %s is listed by the endpoint",
        "usage.?limit",
        "if (( SHARED_MODE || GATEWAY_MODE || OPENROUTER_MODE )); then",
        "TWO ARMS, ONE LAUNCHER",
        "SERIAL CHAIN",
    ):
        assert token in text, token
    assert 'if (( ! SHARED_MODE )); then' not in text
    assert 'export OPENAI_MODEL="$CLAUDE_MODEL"' not in text
    assert '--model "$CLAUDE_MODEL"' not in text
    # Never echo the gateway URL, only its host.
    assert 'echo "using $ARM_LABEL gateway endpoint at $GATEWAY_HOST' in text
    assert "$OPENAI_BASE_URL (" not in text and 'endpoint at $OPENAI_BASE_URL' not in text
    # The marker check skips under FATAL_POLICY=skip and still exits otherwise.
    marker = text.index('if [[ -e "$FATAL_MARKER" ]]')
    block = text[marker : text.index('"$OUT_DIR/campaign_report.json"', marker)]
    assert 'if [[ "$FATAL_POLICY" == "skip" ]]; then' in block
    assert "continue" in block and "exit 1" in block
    # Six in-loop fatal branches break out of the retry loop under skip (four
    # since v3.6, two of the 2026-11-06 reasoning-saturation rule); the
    # validator branch continues to the next point; none is a bare exit any more.
    assert text.count("fatal_link_exit; break") == 6
    assert text.count("fatal_link_exit; continue") == 1
    # A capped quota wait ends the link instead of starting the next point.
    assert "QUOTA_GIVE_UP=1" in text
    give_up = text.index("if (( QUOTA_GIVE_UP )); then")
    assert give_up < text.index("if (( SECONDS - JOB_START > WALL_S - WALL_RESERVE_S )); then", give_up)


def test_v36_serial_submitter_pins_one_chain_over_the_grid() -> None:
    text = SUBMIT_GPT.read_text(encoding="utf-8")
    for token in (
        'POINTS="${POINTS:-360 288 260 240 220 200 180 160 140 120 98 80 60}"',
        'CHAIN_LINKS="${CHAIN_LINKS:-20}"',
        'RUN_TAG="${RUN_TAG:-v37}"',
        'PROJ="${PROJ:-}"',
        'set PROJ=<PROJECT> (the PBS project to charge)',
        'EXTEND="${EXTEND:-0}"',
        'FATAL_POLICY="${FATAL_POLICY:-skip}"',
        'JOB_NAME="omni_gpt_v36"',
        "nscc_claude_grid_v36.pbs",
        'POINTS_JOINED="$(IFS=:; echo "${points[*]}")"',
        'common_vars="ARM=gpt,POINTS=$POINTS_JOINED,FATAL_POLICY=$FATAL_POLICY,RUN_TAG=$RUN_TAG,TURNS=$TURNS,VALIDATE_FORMAL=$VALIDATE_FORMAL,REPO_ROOT=$REPO_ROOT"',
        'qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -W "depend=afterany:$previous" -v "$common_vars" "$PBS_SCRIPT"',
        'NCPUS="${NCPUS:-128}"',
        'AFTER="${AFTER:-}"',
        "note: NCPUS=$NCPUS differs from the worker count these points last ran with",
        "refusing a second chain over the same list",
        "shares a point with",
        '"$RUN_TAG" != "v37"',
        "GPT_MODEL",
        "GPT_ENDPOINT_FILE",
        "QUOTA_MAX_WAIT_S",
    ):
        assert token in text, token
    header = text[: text.index("set -euo pipefail")]
    assert "WHY SERIAL" in header and "EXTEND=1" in header and "100 jobs" in header
    assert "TWO CHAINS = TWO CAMPAIGNS AT A TIME" in header
    assert 'POINTS="360 260 220 180 140 98 60" CHAIN_LINKS=20' in header
    assert 'POINTS="288 240 200 160 120 80"    CHAIN_LINKS=15' in header
    assert "a point may live in one chain only" in text
    assert "POINTS=144 TURNS=2 RUN_TAG=v37_canary VALIDATE_FORMAL=0 CHAIN_LINKS=1" in header
    wrapper = STATUS_GPT.read_text(encoding="utf-8")
    assert 'ARM=gpt exec bash "$(dirname "${BASH_SOURCE[0]}")/claude_v36_status.sh" "$@"' in wrapper


_STUB_QSUB = """#!/bin/bash
n=$(cat "$QSUB_COUNTER" 2>/dev/null || echo 100); n=$((n + 1)); echo "$n" > "$QSUB_COUNTER"
printf '%s\n' "$*" >> "$QSUB_LOG"
echo "$n.pbs101"
"""

_STUB_QSTAT = """#!/bin/bash
if [[ "$1" == "-u" ]]; then
  for j in $ACTIVE_GPT; do printf '%s user q4 omni_gpt_v3 1 1 128 100gb 24:00 H 00:00\n' "$j"; done
  exit 0
fi
if [[ "$1" == "-f" ]]; then
  echo "Job Id: $2"
  echo "    Job_Name = omni_gpt_v36"
  echo "    Variable_List = ARM=gpt,POINTS=${ACTIVE_POINTS:-360:288},FATAL_POLICY=skip,RUN_TAG=v37,TURNS=30,VALIDATE_FORMAL=1"
  exit 0
fi
exit 1
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_v36_serial_submitter_chains_links_and_extends_without_a_gap(tmp_path: Path) -> None:
    """Against stub qsub/qstat: one chain of N afterany links carrying the
    whole colon-joined list, -N omni_gpt_v36 and -P on every link; a live
    chain is refused without EXTEND=1 and appended after its highest link with it."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "qsub").write_text(_STUB_QSUB, encoding="utf-8")
    (bin_dir / "qstat").write_text(_STUB_QSTAT, encoding="utf-8")
    for stub in ("qsub", "qstat"):
        (bin_dir / stub).chmod(0o755)
    log = tmp_path / "qsub.log"
    counter = tmp_path / "counter"

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update({
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "REPO_ROOT": str(REPO_ROOT),
            # Never let the frozen-worker pre-check read the real runs/ tree
            # (on a cluster it holds live campaigns); the test owns OUT_ROOT.
            "OUT_ROOT": str(tmp_path / "runs_default"),
            "QSUB_LOG": str(log),
            "QSUB_COUNTER": str(counter),
            "ACTIVE_GPT": "",
            "USER": env.get("USER", "tester"),
            "PROJ": "testproj",
        })
        env.update(extra)
        return subprocess.run(["bash", str(SUBMIT_GPT)], capture_output=True, text=True, env=env)

    proc = run(CHAIN_LINKS="3")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        assert "-N omni_gpt_v36" in line and "-P testproj" in line
        assert "-l select=1:ncpus=128:mem=100gb" in line
        assert "ARM=gpt,POINTS=360:288:260:240:220:200:180:160:140:120:98:80:60,FATAL_POLICY=skip,RUN_TAG=v37,TURNS=30,VALIDATE_FORMAL=1" in line
        assert str(PBS) in line
    assert "depend=" not in lines[0]
    assert "depend=afterany:101.pbs101" in lines[1]
    assert "depend=afterany:102.pbs101" in lines[2]
    assert "chain: 101.pbs101 102.pbs101 103.pbs101" in proc.stdout

    full = "360:288:260:240:220:200:180:160:140:120:98:80:60"
    live = dict(ACTIVE_GPT="101.pbs101 102.pbs101 103.pbs101", ACTIVE_POINTS=full)

    # A live chain over the SAME list is refused ...
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="2", **live)
    assert proc.returncode == 3, proc.stderr + proc.stdout
    assert "refusing a second chain over the same list" in proc.stderr
    assert log.read_text(encoding="utf-8") == ""

    # ... and extended after its highest-numbered link with EXTEND=1.
    proc = run(CHAIN_LINKS="2", EXTEND="1", **live)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "depend=afterany:103.pbs101" in lines[0]
    assert "depend=afterany:104.pbs101" in lines[1]

    # A DIFFERENT list that shares a point with a live chain is refused, with
    # or without EXTEND=1: one point, one chain.
    for extend in ("0", "1"):
        log.write_text("", encoding="utf-8")
        proc = run(CHAIN_LINKS="1", POINTS="288 240", EXTEND=extend, ACTIVE_GPT="201.pbs101", ACTIVE_POINTS="360:288")
        assert proc.returncode == 3, proc.stderr + proc.stdout
        assert "shares a point" in proc.stderr and "201.pbs101:b288" in proc.stderr
        assert log.read_text(encoding="utf-8") == ""

    # A DISJOINT list is an independent parallel chain (no dependency on the live one).
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="2", POINTS="288 240 200 160 120 80", ACTIVE_GPT="201.pbs101 202.pbs101", ACTIVE_POINTS="360:260:220:180:140:98:60")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "depend=" not in lines[0] and "POINTS=288:240:200:160:120:80," in lines[0]
    assert "depend=afterany:" in lines[1] and "afterany:201" not in lines[1] and "afterany:202" not in lines[1]

    # NCPUS=64 requests a q2 node; AFTER=<jobid> chains the first link behind it.
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="2", POINTS="260 220", NCPUS="64", AFTER="777.pbs101")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert "-l select=1:ncpus=64:mem=100gb" in lines[0] and "depend=afterany:777.pbs101" in lines[0]
    assert "depend=afterany:101.pbs101" in lines[1] or "depend=afterany:" in lines[1]
    proc = run(CHAIN_LINKS="1", NCPUS="96")
    assert proc.returncode == 2 and "NCPUS must be 128" in proc.stderr

    # A point that last ran with 127 workers may resume on 64 cores (2026-09-05:
    # slow_workers is a runtime key); the submitter notes it and submits.
    out_root = tmp_path / "runs"
    (out_root / "fig2_gptsub_b360_v37").mkdir(parents=True)
    (out_root / "fig2_gptsub_b360_v37" / "protocol_snapshot.json").write_text('{"slow_workers": 127}', encoding="utf-8")
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="1", POINTS="360 260", NCPUS="64", OUT_ROOT=str(out_root))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "b360: last slow_workers=127 -> now 63 workers" in proc.stdout
    assert "-l select=1:ncpus=64:mem=100gb" in log.read_text(encoding="utf-8")
    proc = run(CHAIN_LINKS="1", POINTS="360 260", NCPUS="128", OUT_ROOT=str(out_root))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "OUT_ROOT=" + str(out_root) in log.read_text(encoding="utf-8")
    assert "REPO_ROOT=" + str(REPO_ROOT) in log.read_text(encoding="utf-8")
    # a second checkout runs the next links against the same runs/ tree
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="1", POINTS="360 260", NCPUS="128", OUT_ROOT=str(out_root), REPO_ROOT=str(tmp_path / "OmniQEC_b"), PBS_SCRIPT=str(PBS))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "REPO_ROOT=" + str(tmp_path / "OmniQEC_b") in log.read_text(encoding="utf-8")

    # Formal submissions keep the protocol tag and turn count; a canary must say so.
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="1", TURNS="2")
    assert proc.returncode == 2 and "formal submission requires RUN_TAG=v37 and TURNS=30" in proc.stderr
    proc = run(CHAIN_LINKS="1", TURNS="2", RUN_TAG="v37_canary", VALIDATE_FORMAL="0", POINTS="144")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "POINTS=144,FATAL_POLICY=skip,RUN_TAG=v37_canary,TURNS=2,VALIDATE_FORMAL=0" in log.read_text(encoding="utf-8")


_STUB_QSTAT_CLAUDE = """#!/bin/bash
# ACTIVE_CLAUDE = "<jid>:<point> ..." — one live omni_claude_v36 link per entry.
if [[ "$1" == "-u" ]]; then
  for e in $ACTIVE_CLAUDE; do printf '%s user q2 omni_claude_v 1 1 64 100gb 24:00 H 00:00\n' "${e%%:*}"; done
  exit 0
fi
if [[ "$1" == "-f" ]]; then
  for e in $ACTIVE_CLAUDE; do
    if [[ "${e%%:*}" == "$2" ]]; then
      echo "Job Id: $2"
      echo "    Job_Name = omni_claude_v36"
      echo "    Variable_List = POINTS=${e##*:},RUN_TAG=v37,TURNS=30,VALIDATE_FORMAL=1"
      exit 0
    fi
  done
fi
exit 1
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_v36_submitter_extends_live_point_chains_without_a_gap(tmp_path: Path) -> None:
    """Against stub qsub/qstat: a point with a live chain is refused without
    EXTEND=1 (nothing submitted); with it the new links hang after that
    point's HIGHEST live link while a point with no live chain starts fresh."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "qsub").write_text(_STUB_QSUB, encoding="utf-8")
    (bin_dir / "qstat").write_text(_STUB_QSTAT_CLAUDE, encoding="utf-8")
    for stub in ("qsub", "qstat"):
        (bin_dir / stub).chmod(0o755)
    log = tmp_path / "qsub.log"
    counter = tmp_path / "counter"

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update({
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "REPO_ROOT": str(REPO_ROOT),
            "OUT_ROOT": str(tmp_path / "runs_default"),
            "QSUB_LOG": str(log),
            "QSUB_COUNTER": str(counter),
            "ACTIVE_CLAUDE": "",
            "USER": env.get("USER", "tester"),
            "PROJ": "testproj",
        })
        env.update(extra)
        return subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env)

    # Fresh grid: one independent chain per point, links chained by afterany.
    proc = run(CHAIN_LINKS="2", POINTS="360 288", NCPUS="64")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert "POINTS=360,RUN_TAG=v37,TURNS=30,VALIDATE_FORMAL=1" in lines[0] and "depend=" not in lines[0]
    assert "depend=afterany:101.pbs101" in lines[1]
    assert "POINTS=288," in lines[2] and "depend=" not in lines[2]
    assert "depend=afterany:103.pbs101" in lines[3]
    assert all("-l select=1:ncpus=64:mem=100gb" in line for line in lines)

    # b140 has live links 301 (running) and 302 (held); b120 has none.
    live = dict(ACTIVE_CLAUDE="302.pbs101:140 301.pbs101:140 401.pbs101:98")
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="3", POINTS="140 120", NCPUS="64", **live)
    assert proc.returncode == 3, proc.stderr + proc.stdout
    assert "refusing duplicate Claude v3.6 submission" in proc.stderr
    assert "301.pbs101:v37:b140" in proc.stderr and "302.pbs101:v37:b140" in proc.stderr
    assert "b98" not in proc.stderr and "EXTEND=1" in proc.stderr
    assert log.read_text(encoding="utf-8") == ""

    proc = run(CHAIN_LINKS="3", POINTS="140 120", NCPUS="64", EXTEND="1", **live)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    # b140: three links appended after the highest live id 302, then each other.
    assert "POINTS=140," in lines[0] and "depend=afterany:302.pbs101" in lines[0]
    assert "depend=afterany:105.pbs101" in lines[1] and "depend=afterany:106.pbs101" in lines[2]
    # b120: fresh chain, no dependency on anything live.
    assert "POINTS=120," in lines[3] and "depend=" not in lines[3]
    assert "depend=afterany:108.pbs101" in lines[4]
    assert "b140: extending the live chain, first new link depends on afterany:302.pbs101" in proc.stdout
    assert "b120: no live chain under RUN_TAG=v37" in proc.stdout

    # EXTEND=1 onto 128 cores for a point that last ran with 63 workers: a note,
    # and the new links carry NCPUS=128 (2026-09-05: slow_workers is a runtime key).
    out_root = tmp_path / "runs"
    (out_root / "fig2_claudesub_b140_v37").mkdir(parents=True)
    (out_root / "fig2_claudesub_b140_v37" / "protocol_snapshot.json").write_text('{"slow_workers": 63}', encoding="utf-8")
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="1", POINTS="140", NCPUS="128", EXTEND="1", OUT_ROOT=str(out_root), **live)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "b140: last slow_workers=63 -> now 127 workers" in proc.stdout
    logged = log.read_text(encoding="utf-8")
    assert "depend=afterany:302.pbs101" in logged and "-l select=1:ncpus=128:mem=100gb" in logged
    assert "REPO_ROOT=" + str(REPO_ROOT) in logged
    proc = run(CHAIN_LINKS="1", POINTS="140", EXTEND="2")
    assert proc.returncode == 2 and "EXTEND must be 0 or 1" in proc.stderr
    # No project, no submission: PROJ has no account-derived default.
    log.write_text("", encoding="utf-8")
    proc = run(CHAIN_LINKS="1", POINTS="140", PROJ="")
    assert proc.returncode == 2 and "set PROJ=<PROJECT>" in proc.stderr
    assert log.read_text(encoding="utf-8") == ""


# --- DeepSeek arm: the same launcher, ARM=deepseek through OpenRouter (2026-09-10)

def test_v36_launcher_runs_the_deepseek_arm_through_openrouter() -> None:
    """ARM=deepseek is a pure HTTP client of OpenRouter: no CLIProxyAPI binary,
    no OAuth credential, no shared endpoint; the key comes from
    ~/.config/omniqec/openrouter.env, the model id is asserted against the
    endpoint's /v1/models, the run dir / log prefix use the DeepSeek arm name, and
    run-campaign / the validator get --provider openrouter."""

    text = PBS.read_text(encoding="utf-8")
    for token in (
        'DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek/deepseek-v4-pro}"',
        # 2026-10-20: the prefix/label default to the DeepSeek arm's; OPENROUTER_ARM_PREFIX/_LABEL rename a comparison arm
        'deepseek) ARM_PREFIX="${OPENROUTER_ARM_PREFIX:-deepseek}"; ARM_LABEL="${OPENROUTER_ARM_LABEL:-DeepSeek}"; LLM_MODEL="$DEEPSEEK_MODEL"; PROVIDER="openrouter" ;;',
        'claude) ARM_PREFIX="claudesub"; ARM_LABEL="Claude";   LLM_MODEL="$CLAUDE_MODEL"; PROVIDER="openai" ;;',
        'OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-$HOME/.config/omniqec/openrouter.env}"',
        'if [[ "$ARM" == "deepseek" ]]; then',
        'OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"',
        '[[ "$OPENROUTER_BASE_URL" == */v1 ]]',
        'export OPENROUTER_MODEL="$LLM_MODEL"',
        'MODELS_URL="$OPENROUTER_BASE_URL/models"',
        'if [[ "$ARM" != "claude" ]]; then',
        'models_listed="$(curl -sf --max-time 20 ${CURL_RESOLVE[@]+"${CURL_RESOLVE[@]}"} "$MODELS_URL" -H "Authorization: Bearer $API_KEY"',
        'OMNIQEC_DNS_PINS',
        "transient gateway failure, not a verdict",
        '--provider "$PROVIDER" \\',
        '--provider "$PROVIDER" --model "$LLM_MODEL"',
        'python -m omniqec_agentic.hpc.probe_llm "$PROVIDER" > /dev/null 2>&1',
        "insufficient.?credits",
        "THIRD ARM",
    ):
        assert token in text, token
    # The DeepSeek arm never touches the CLIProxyAPI paths ...
    assert 'if (( ! SHARED_MODE && ! GATEWAY_MODE && ! OPENROUTER_MODE )); then' in text
    assert 'if (( ! GATEWAY_MODE && ! OPENROUTER_MODE )) && [[ -f "$SHARED_ENDPOINT_FILE" ]]; then' in text
    # ... and the OpenRouter URL (which may carry the Worker's path token) is
    # never echoed, only its host.
    assert 'echo "using OpenRouter endpoint at $OPENROUTER_HOST' in text
    assert 'endpoint at $OPENROUTER_BASE_URL' not in text
    assert "python -m omniqec_agentic.hpc.probe_llm openai" not in text
    assert "--provider openai" not in text


def test_v36_submitter_and_status_take_the_deepseek_arm() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    assert 'ARM="${ARM:-claude}"' in text
    assert 'deepseek) ARM_PREFIX="deepseek";  ARM_LABEL="DeepSeek" ;;' in text
    assert 'JOB_NAME="omni_${ARM}_v36"' in text
    assert '[[ "$ARM" == "claude" ]] || common_vars="ARM=$ARM,$common_vars"' in text
    assert "DEEPSEEK_MODEL OPENROUTER_ENV_FILE" in text
    assert "awk '$0 ~ /omni_/ {print $1}'" in text
    wrapper = STATUS_DEEPSEEK.read_text(encoding="utf-8")
    assert 'ARM=deepseek exec bash "$(dirname "${BASH_SOURCE[0]}")/claude_v36_status.sh" "$@"' in wrapper


def test_v36_submitter_submits_deepseek_chains_under_their_own_job_name(tmp_path: Path) -> None:
    """Against stub qsub/qstat: ARM=deepseek passes ARM in -v, names the jobs
    omni_deepseek_v36, and a live Claude chain on the same point is NOT a
    duplicate of it (different arm, different run dir)."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "qsub").write_text(_STUB_QSUB, encoding="utf-8")
    (bin_dir / "qstat").write_text(_STUB_QSTAT_CLAUDE, encoding="utf-8")
    for stub in ("qsub", "qstat"):
        (bin_dir / stub).chmod(0o755)
    log = tmp_path / "qsub.log"
    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{env.get('PATH', '')}",
        "REPO_ROOT": str(REPO_ROOT),
        "OUT_ROOT": str(tmp_path / "runs"),
        "QSUB_LOG": str(log),
        "QSUB_COUNTER": str(tmp_path / "counter"),
        "ACTIVE_CLAUDE": "301.pbs101:144",   # a live CLAUDE chain on b144
        "USER": env.get("USER", "tester"),
        "PROJ": "testproj",
        "ARM": "deepseek", "CHAIN_LINKS": "2", "POINTS": "144 98", "NCPUS": "128",
    })
    proc = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert "-N omni_deepseek_v36" in lines[0]
    assert "POINTS=144,ARM=deepseek,RUN_TAG=v37,TURNS=30,VALIDATE_FORMAL=1" in lines[0]
    assert "depend=afterany:101.pbs101" in lines[1]
    assert "POINTS=98,ARM=deepseek," in lines[2]
    assert "DeepSeek chains" in proc.stdout and "ARM=deepseek" in proc.stdout
    env["ARM"] = "gpt"
    proc = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env)
    assert proc.returncode == 2 and "ARM must be claude or deepseek" in proc.stderr


def test_launcher_quota_heuristic_ignores_digits_inside_numbers(tmp_path) -> None:
    """2026-09-12: `429` inside plpr=0.01638896824911429 (a [slow] line of
    the attempt log) sent a strict-turn abort into a 15-minute quota poll."""

    import re
    import subprocess

    text = PBS.read_text(encoding="utf-8")
    match = re.search(r"looks_like_quota\(\) \{\n(.*?)\n\}", text, re.S)
    assert match, "looks_like_quota not found"
    fn = "looks_like_quota() {\n" + match.group(1) + "\n}"
    cases = {
        "  [slow] g0001_p0004 [[28,2,5]] bb ok plpr=0.01638896824911429 by=telescoping_cpu\n": False,
        "plpr=0.429 shots=4290\n": False,
        "family_quota: {'bb': 6}\n": False,
        "openai.RateLimitError: Error code: 429 - too many requests\n": True,
        "InternalServerError: Error code: 503 - auth_unavailable: no auth available\n": True,
        "HTTP 429 returned by the gateway\n": True,
        "quota exceeded for this window\n": True,
        "insufficient_credits: top up\n": True,
    }
    for body, expected in cases.items():
        probe = tmp_path / "attempt.log"
        probe.write_text(body, encoding="utf-8")
        rc = subprocess.run(["bash", "-c", fn + f"\nlooks_like_quota '{probe}'"], capture_output=True).returncode
        assert (rc == 0) is expected, (body, rc)


def test_launchers_never_reference_shard_split_without_a_default():
    """The launcher runs under ``set -u``; with persistent workers on, OMNIQEC_SHARD_SPLIT stays
    unset, so every reference must carry a default (2026-09-14: a bare ``$OMNIQEC_SHARD_SPLIT`` in the
    banner killed every new-tree link at start-up)."""
    import re
    for name in ("nscc_claude_grid_v36.pbs",):
        text = (HPC / name).read_text(encoding="utf-8")
        bare = [m.group(0) for m in re.finditer(r"\$\{?OMNIQEC_SHARD_SPLIT\b(?![:}]?[:-])[^\n]{0,12}", text)
                if not m.group(0).startswith("${OMNIQEC_SHARD_SPLIT:-")]
        assert bare == [], (name, bare)

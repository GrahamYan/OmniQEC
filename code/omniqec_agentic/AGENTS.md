# Repository Guidelines

## Orientation

`omniqec_agentic` is an LLM-driven stabilizer-code discovery framework. Before
changing anything, read [`docs/04_architecture.md`](docs/04_architecture.md) —
the as-built snapshot of what actually runs. `docs/01`–`docs/03a` are the frozen
design specs behind it. The authority order is explicit and load-bearing:
**code > `docs/04` > design specs**. A finalized design spec is amended by
appending an 实施修订 section, never by rewriting its body; deviations that
persist belong in `docs/04` §11.

## Project Structure & Module Organization

`omniqec_agentic` is a Python package; run package commands from its parent
directory. The layering and its dependency rule are specified in
`docs/04_architecture.md` §1 — the short version:

- `workflow/` — the LangGraph campaign graph, state, protocol snapshot, and
  SQLite checkpoints. **The only package that may import langgraph.** Swapping
  frameworks must mean rewriting this layer alone.
- `orchestrator/` — the policy layer (framework-free): the LLM tool loop
  (`loop.py`), the `TurnPlan` schema and its validator (`plan.py`), the tool
  registry (`registry.py`), the evolving manual (`manual.py`), the per-turn
  reflector, versioned prompts under `prompts/`, and `policies/` — home of
  `FallbackPolicy`, the deterministic degradation path that `make_policy()`
  in `__init__.py` selects when the provider is `dry_run`.
- `tools/` — the policy-agent tool suite in `read` / `draft` / `policy` scopes,
  not a utility drawer. A function becomes a tool by carrying `@tool` from
  `orchestrator.registry`, and its module must appear in
  `registry._ensure_tools_loaded` to be imported at all. The decorated
  function's docstring is rendered verbatim into the model's tool catalog
  (`manual/static/tools.md` and the inline S-1 context block), so **treat every
  tool-docstring edit as a prompt change.**
- `construction/` — the four family plugins (`bb`, `hgp`, `lp.py` serving both
  `lp_abelian` and `lp_nonabelian`), the group catalog under `groups/`, seed
  grammars under `grammars/`, and the coarse structure coordinates
  (`structure.py`) shared by the coverage meter, evidence rows, and ledger
  predicates.
- `sec/` — syndrome-extraction circuit synthesis: sector graphs, deterministic
  edge coloring, schedule ranking, the Stim builder and its per-profile
  lowerings, and the circuit audits.
- `verification/` — circuits, decoders, DEM, distance channels, noise profiles,
  proxy distance, and structural checks.
- `evaluation/` — the framework-free evaluator library: single-candidate
  pipeline, work-stealing process pool, DEM cache, anchor-p racing, evidence
  merging, writeback, and slow-loop feedback.
- `memory/` — campaign memory and the learning layer: the MAP-Elites archive,
  the insight ledger, the deterministic prospective backtester (the ledger's
  only judge of a hypothesis), the UCB1 strategy bandit, two-tier Tanner
  fingerprints, the known-code table, and structural coverage.
- `llm/` — completion-only provider clients (deepseek / openrouter / claude /
  openai_compat) plus timeout, retry, and usage normalization. No langchain
  model integrations.
- `deformation/` — run-level Hadamard-mask dispatch (`sector_h`), default off
  and deliberately invisible to the LLM.

Package-root modules include `cli.py` (13 subcommands), `funsearch.py` (the fast
FunSearch loop and the Generator prompt assembly), `campaign.py` (the fast+slow
glue), `ranking.py`, `config.py` (frozen config dataclasses and protocol knobs),
`sandbox.py` (restricted `construct(rng)` execution), `qualification.py`,
`schemas.py`, `analyze.py`, `export_stim.py`, `errors.py`, and the exporters
`progress_csv.py` / `summary_csv.py` / `results_csv.py` / `token_usage.py`.

The paper-preset PBS launchers, generic tools, and the LLM gateway live in `hpc/`
(site-specific deployment scripts and runbooks are not part of the open-source tree); pytest
coverage in `tests/` (with pinned golden circuits in `tests/golden/`); design
docs and the versioned qualification set in `docs/`. Treat `runs/` (at the repo
root, above this package) as generated experiment output and `tmp/` as reference
or scratch data, not production modules.

## Build, Test, and Development Commands

Python 3.12; dependency versions are pinned and `ldpc==2.4.1` must be exact.
From the directory containing `omniqec_agentic/`:

```bash
conda create -n omniqec python=3.12 -y && conda activate omniqec
pip install -r omniqec_agentic/hpc/requirements_fig2_nscc.txt
python -m pytest omniqec_agentic/tests -q
python -m py_compile $(find omniqec_agentic -name '*.py' | sort)
python -m omniqec_agentic.cli --help
```

`hpc/requirements_fig2_nscc.txt` is the working dependency set — the older
package-root `requirements.txt` predates the workflow layer and pins neither
pydantic nor langgraph. The primary decoder (beam search, vendored C++/Cython)
is compiled in place rather than pip-installed; see the repo-root `README.md`
§0 for that build and the `.pth` step.

Before changing construction, scoring, sandboxing, SEC, or verification, run the
suite plus a keyless end-to-end smoke — no API key required, since
`--provider dry_run` selects the deterministic `FallbackPolicy`:

```bash
python -m pytest omniqec_agentic/tests -q
python -m omniqec_agentic.cli build-dem-smoke --seed-index 1 --proxy-iters 120
python -m omniqec_agentic.cli run-campaign --budget 72 --out runs/smoke --turns 1 --proposal-count 2 --slow-workers 0
```

`build-dem-smoke` walks construction → SEC schedule → noisy Stim circuit → DEM
and prints a stable `noisy_circuit_sha256`, which makes it usable as a real
regression check. For SEC or lowering changes specifically, the golden
segment-by-segment comparison against the SBB reference circuits is the guard:
`python -m pytest omniqec_agentic/tests/test_sec_golden.py -q`.
`check-slow-deps` reports which optional decoder backends resolved in the
current environment.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation,
`from __future__ import annotations`, type hints on public interfaces, and short
module docstrings that cite the governing design section (e.g. `03 §4`). Use
`snake_case` for functions, variables, and modules; `PascalCase` for classes and
enums; and `UPPER_SNAKE_CASE` for constants. Prefer frozen dataclasses and
explicit serialization methods for persisted configuration. No formatter or
linter is configured, so keep imports grouped, lines readable, and changes
consistent with adjacent code.

Two structural rules are enforced by review, not by tooling:

- only `workflow/` may import langgraph (pydantic is allowed everywhere);
- the policy/protocol split — the LLM decides *who* gets measured and *how much*
  is spent; measurement protocol (SEC synthesis, noise, decoders, statistics, CI
  and hypothesis adjudication) is frozen and has no write interface exposed to
  the agent. Tool return values must be machine-computed facts; an agent's own
  words must never become a record.

## Testing Guidelines

Tests use `pytest`, including `parametrize`, `monkeypatch`, `tmp_path`, and
NumPy assertions; the suite is 159 tests and runs in about 75 seconds. Name
files `test_<feature>.py` and tests `test_<behavior>`. Add regression coverage
beside the affected subsystem, especially for schema migrations, deterministic
seeds, code hashes, CLI defaults, protocol-snapshot fields, and noise-profile
semantics. Avoid network calls and large shot counts in unit tests.

Changing `orchestrator/prompts/*.md` or any field returned by
`build_protocol_snapshot()` changes the protocol snapshot and therefore
**invalidates resume for in-flight campaigns**. Say so in the PR.

Two things that change what the model sees are **not** caught by that check:
tool docstrings (rendered into `manual/static/tools.md` and the inline S-1
context block) and the Generator prompt, which is assembled in code inside
`funsearch.py`. `prompt_ids()` hashes only `orchestrator/prompts/*.md`, so
`--resume` will happily continue a campaign under a silently different agent
context — start a fresh run yourself after touching either. This is the
comparability gap recorded as docs/04 §11 R1.

## Commit & Pull Request Guidelines

Use concise, imperative subjects with an optional scope, for example
`verification: preserve Helios scale metadata`. Keep commits focused. Pull
requests should explain the behavioral or protocol change, list commands run,
link the relevant issue or design section, and call out schema, artifact, or
resume compatibility. Include representative JSON/CSV output for artifact
changes and smoke-job evidence for PBS/Slurm updates.

## Security & Configuration

Keep API keys only in a local `.env` or in `~/.config/omniqec/*.env` (chmod
600); never commit credentials or generated provider transcripts containing
secrets. Preserve deterministic seeds and pinned dependency versions when
comparing experimental results — a campaign in progress must not have its
dependencies upgraded, and a differing protocol snapshot is refused on
`--resume` by design.

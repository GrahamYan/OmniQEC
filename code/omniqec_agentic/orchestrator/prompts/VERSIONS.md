# Prompt versions (human-readable alias table)

The audit record only ever trusts the content hash (`prompt_ids` in the
protocol snapshot); this table is for people. Regenerate hashes with:
`python -c "from omniqec_agentic.orchestrator.prompts import prompt_ids; print(prompt_ids())"`

| name@semver | hash8 | note |
|---|---|---|
| reflector@1.0.0 | (see prompt_ids()) | per-turn qualitative review; advisory only, expires under the S-10 decay window; quantitative claims must still go through ledger_propose + prospective backtesting |
| orchestrator_system@1.0.0 | (see prompt_ids()) | 03a §3.1 v1: neutral one-line identity, 6 tool-verifiable rules, JSON tool-loop contract |

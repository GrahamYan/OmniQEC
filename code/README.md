# OmniQEC code

Source code of OmniQEC, an LLM-guided framework for discovering quantum error-correcting codes and evaluating them in circuit-level quantum-memory and transversal-CNOT experiments.

## Layout

```text
omniqec_agentic/   the Python package (campaign workflow, code constructions, fast and slow evaluation loops,
                   final stage, BO and random baselines, tests)
omniqec_agentic/hpc/   PBS launchers for the paper's runs and generic tools (validator, final-stage tools)
third_party/       vendored decoders (telescoping-decoder, BeamSearchDecoder) with their licenses
```

## Install

Python 3.10 or newer, and a C compiler for the telescoping decoder.

```bash
pip install -r omniqec_agentic/requirements.txt
pip install -e ./third_party/telescoping-decoder
```

Set the API key of the LLM provider you use as an environment variable (see `.env.openrouter.example` for the OpenRouter route).

## Run

One campaign is one qubit budget N. A dry run without any LLM call:

```bash
python -m omniqec_agentic.cli run-campaign --budget 98 --out runs/b98_dry --turns 2 --provider dry_run
```

Main options of `run-campaign`:

- `--task memory | cnot_dense`: quantum memory or transversal logical CNOT.
- `--provider deepseek | openrouter | claude | openai | dry_run`: the LLM backend.
- `--search-method llm | bo | random`: LLM-guided search, or one of the two baselines.
- `--ablation full | no_slow | no_feedback`: the ablation configurations.
- `--exact-check-weight w`: the fixed-weight CNOT searches.
- `--resume`: continue a campaign from its checkpoint.

`python -m omniqec_agentic.cli run-campaign --help` lists all options.

The settings used in the paper are available as presets of the PBS launcher:

```bash
PRESET=fig2_final PROJ=<PROJECT> bash omniqec_agentic/hpc/submit_claude_v36_grid.sh   # quantum memory
PRESET=fig3_final PROJ=<PROJECT> bash omniqec_agentic/hpc/submit_claude_v36_grid.sh   # transversal CNOT
```

The GPT backend is submitted with `omniqec_agentic/hpc/submit_gpt_v36_serial.sh`, and the BO and random baselines with `PROJECT=<PROJECT> bash omniqec_agentic/hpc/submit_baseline_grid.sh`. The header of each script lists its variables; the queue, module and environment names there are those of the cluster used for the paper and may need adjusting.

## Tests

```bash
python -m pytest omniqec_agentic/tests -q
```

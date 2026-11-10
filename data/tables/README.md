# OmniQEC appendix tables

Seven appendix tables, each as a `.tex` file (a `longtblr` in the same format as Tables IX and X of the paper, ready for `\input`) and a `.csv` file with the same content as plain text.

## Tables

| File | LaTeX label | Content |
| --- | --- | --- |
| `tab_bo_memory_top3` | `tab:bo-memory-top3` | BO baseline (Bayesian optimization without an LLM), quantum memory: top-three codes per qubit budget N |
| `tab_bo_cnot_top3` | `tab:bo-cnot-top3` | BO baseline, transversal logical CNOT: top-three codes per qubit budget N |
| `tab_fig4a_cross` | `tab:fig4a-cross-evaluation` | Reciprocal evaluation (Fig. 4a): the memory champion and the CNOT champion of each N and LLM backend, each evaluated under both tasks |
| `tab_fig4c_fixed_weight` | `tab:fig4c-fixed-weight` | Fixed-weight transversal-CNOT searches (Fig. 4c, w = 4–7): top-three codes and Top-3 mean per (N, w) |
| `tab_ablation` | `tab:ablation` | Ablation: Full, w/o slow, w/o slow & fast and Random configurations; leading code and Top-3 mean per cell |
| `tab_reference_codes` | `tab:bb-reference-codes` | Measurements of the two BB reference codes ([[72,12,6]] at N = 144, [[144,12,12]] at N = 288) |
| `tab_champion_stats` | `tab:champion-statistics` | Measurement statistics of the rank-1 codes of Tables IX and X |

## Columns

- **N**: physical-qubit budget; a code satisfies n + m_X + m_Z ≤ N. In the CNOT task N is the budget per code block.
- **Rank**: rank within the cell.
- **Task / Objective**: Memory or CNOT. In `tab_fig4a_cross`, Objective is the task of the search that produced the code.
- **Backend**: LLM backend (Claude, DeepSeek, GPT). **Configuration** (`tab_ablation`): the ablation configuration as named in the paper.
- **Code** (`tab_reference_codes`): reference code name and parameters [[n,k,d]].
- **n,k,d**: block length, number of logical qubits and the distance estimate used during search.
- **Family**: construction family (BB, HGP, LP Abelian, LP Non-Abelian).
- **G, A, B**: the group and the two group-algebra elements of the construction.
- **w**: maximum check weight (largest row weight of H_X and H_Z).
- **F_X, F_Z**: observed block failures in the X and Z bases.
- **p_LER**: logical error rate per logical qubit and per round, as defined in the paper. In `tab_fig4a_cross` the two p_LER columns are the values under the memory and the CNOT task.
- **95% interval**: Wilson intervals (z = 1.96) per basis, propagated to p_LER. In `tab_fig4a_cross` they are written as `[a, b]×10^e`, with the exponent of the p_LER in the same cell.
- **Top-3 mean**: arithmetic mean of the p_LER of the three leading codes (in `tab_fig4c_fixed_weight` given on the rank-1 row).
- **†**: a p_LER from pooled measurements, as defined in each table caption. In `tab_fig4a_cross`, bold marks the lower of the two champions' values in a task column for the same (N, backend).

## CSV format

- UTF-8. The header is the plain-text `.tex` header and the columns match the `.tex` columns one to one, except that 95% interval columns are split into `… low` and `… high`.
- Numbers keep the three significant figures of the tables, written as `8.37e-04`; the `†` marker is kept as a suffix (e.g. `2.99e-05†`). Bold and thousands separators are not kept.
- `n,k,d` is written as `30,2,5`. Groups are written as `C3×C5` and polynomials as `1 + x^2 y^2` (a space denotes a product). In `tab_fig4c_fixed_weight` the Top-3 mean is empty on the rank-2 and rank-3 rows.

`MANIFEST.md5` lists the MD5 checksum of every file in this directory.

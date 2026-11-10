# OmniQEC code records

One JSON record per distinct quantum error-correcting code listed in the tables of the paper. Records are keyed by `code_hash`; a code that is listed in several tables, or in several rows of one table, has one record with one `appearances` entry per table row.

## Tables covered

| `table` value | Content |
| --- | --- |
| `Table IX` | Top-three constructions from the quantum-memory experiments, per qubit budget N and LLM backend |
| `Table X` | Top-three constructions from the transversal logical-CNOT experiments, per qubit budget N and LLM backend |
| `tab_bo_memory_top3` | Top-three constructions from the BO baseline, quantum memory |
| `tab_bo_cnot_top3` | Top-three constructions from the BO baseline, transversal logical CNOT |
| `tab_fig4a_cross` | Reciprocal evaluation: each memory champion and each CNOT champion evaluated under both tasks |
| `tab_champion_stats` | Measurement statistics of the champion codes of Tables IX and X |
| `tab_fig4c_fixed_weight` | Top-three constructions from the fixed-weight transversal logical-CNOT searches |
| `tab_ablation` | Ablation configurations (Full, w/o slow, w/o slow & fast, Random): the leading construction of each cell, and the three codes whose values form the printed Top-3 mean |
| `tab_reference_codes` | The two BB reference codes, [[72,12,6]] and [[144,12,12]] |

## Files

| File | Content |
| --- | --- |
| `records/<code_hash>.json` | One record per code (fields below) |
| `index.csv` | One row per code: `code_hash`, `code_hash8`, `family`, `n`, `k`, `d`, `w`, `tables` (the tables it appears in, separated by `;`), `n_appearances` |
| `MANIFEST.md5` | MD5 checksum of every file, one `<md5> <path>` line per file |

## Record fields

- `code_hash`: 24 hexadecimal digits. The first 24 digits of the SHA-256 digest of, in order: the shape of H_X written as a Python tuple string (for example `(15, 30)`), the entries of H_X as unsigned 8-bit integers in row-major order, the shape string of H_Z, and the entries of H_Z. Two records with the same `code_hash` have identical H_X and H_Z. This is `omniqec_agentic.construction.build.compiled_code_hash` in `code/`.
- `code_hash8`: the first 8 digits of `code_hash`, as used in figure labels.
- `family`: the construction family as labelled in the tables: `BB`, `HGP`, `LP Abelian` or `LP Non-Abelian`. It names the construction form of `spec`; it is not a classification up to code equivalence.
- `n`, `k`, `d`: the code parameters [[n,k,d]] as printed in the tables. `n` and `k` equal the block length and the number of logical qubits of H_X and H_Z. `d` is the distance estimate printed in the tables (the estimate used during search).
- `parameters_printed_in_tables`: `false` for codes whose [[n,k,d]] no table prints (codes listed only as rank 2 or 3 of `tab_ablation`); their `d` is the same search-stage distance estimate, taken from the archived evaluation.
- `w`: the maximum check weight, i.e. the largest row weight of H_X and H_Z.
- `construction`:
  - `G`, `A`, `B`: the group and the two algebra elements in the notation of the tables.
  - `as_printed_in`: the tables that print this construction. When it is empty, the code appears only in tables without construction columns, `rendered_from_spec` is `true`, and `G`, `A`, `B` are written from `spec` in the same notation.
  - `spec`: the machine-readable construction. `omniqec_agentic.construction.build.build_matrices(spec)` gives the stored H_X and H_Z and hence `code_hash`.
- `parity_check_matrices`: `Hx` and `Hz`, each with `shape` (`[number of checks, n]`) and `row_supports` (for each row, the 0-based column indices of its nonzero entries). All matrices are over GF(2) and satisfy H_X H_Z^T = 0.
- `appearances`: one entry per table row in which the code is listed:
  - `table`: one of the values in the table above.
  - `task`: `Memory` or `CNOT`. For `tab_fig4a_cross` it is the Objective column, i.e. the task of the search that produced the code.
  - `N`: the qubit budget column of the table. In the CNOT tables N is the budget per code block.
  - `backend_or_method`: the Backend, Method or Configuration column as the table states it; `fixed weight w=<w>` for `tab_fig4c_fixed_weight`; the Code column (`BB [[72,12,6]]`, `BB [[144,12,12]]`) for `tab_reference_codes`.
  - `rank`: the Rank column. It is 1 in tables that list one champion per cell (`tab_fig4a_cross`, `tab_champion_stats`) and `null` in `tab_reference_codes`. In `tab_ablation`, rank 1 is the printed leading construction and ranks 2 and 3 are the other two codes whose values enter the printed Top-3 mean; the mean of the three `p_LER` values of a cell is the printed Top-3 mean.
  - `p_LER`: the logical error rate exactly as the table prints it, written as `m.mme±XX`, with the table's markers: a trailing `†` marks a value from pooled terminal and supplementary measurements; a leading `<` with a trailing `§` marks an upper bound from a zero-failure measurement.
  - `p_LER_value`: the printed value as a number (for an upper bound, the bound).
  - `pooled`: `true` when the table marks the value with `†`.
  - `upper_bound`: `true` when the table prints the value as an upper bound (`<`, `§`).
  - `ci95`: the 95% interval `[low, high]` of the same measurement, to three significant figures as in the tables.
  - `ci95_printed_in_table`: `true` when the table prints this interval. Tables IX and X, `tab_fig4c_fixed_weight` and `tab_ablation` print only the value; their intervals belong to the measurement behind the printed value.
  - `p_LER_printed_in_table`: `true` when the table prints this value. It is `false` only for ranks 2 and 3 of `tab_ablation`, whose values appear in the table through the Top-3 mean; they are given to three significant figures.
  - `p_LER_in_top3_mean` (only in `tab_ablation` entries, ranks 1 to 3): the value entering the printed Top-3 mean, at the precision of the archived value; to three significant figures it equals `p_LER`. The mean of the three values of a cell equals the printed Top-3 mean to three significant figures.
  - `p_LER_by_task` (only in `tab_fig4a_cross` entries): the fields `p_LER`, `p_LER_value`, `pooled`, `upper_bound`, `ci95`, `ci95_printed_in_table` and `p_LER_printed_in_table` for both evaluation tasks, `Memory` and `CNOT`, instead of a single value.

Each value follows its own table. The same code can therefore carry different values in different tables or rows; the definition of p_LER and the measurement protocol of each table are given in the paper.

## Construction notation

- `C3×C5` is the direct product of cyclic groups; a single factor `C15` is a cyclic group.
- BB: G = C_ℓ×C_m with commuting generators x (order ℓ) and y (order m). A single factor C_m means ℓ = 1, and only y appears. In `spec`, `ell` and `m` are the two orders and each term `[i, j]` of `A` or `B` is x^i y^j. `"canonicalized": true` marks a BB `spec` whose two axes were ordered so that ℓ ≤ m.
- LP Abelian: the generators x, y, z follow the order of the cyclic factors of G. In `spec`, `group.factors` lists the factor orders; `A` and `B` are matrices whose entries are lists of terms, each term an exponent vector over the generators.
- LP Non-Abelian: `C11:C4(10)` denotes the group with generators r, s and relations r^11 = s^4 = 1, s^-1 r s = r^10; each term `[i, j]` of `spec` is r^i s^j. `C3×D7` denotes the direct product of C_3, generated by the central element c, with the dihedral group D_7, generated by r and s with r^7 = s^2 = 1, s r s = r^-1; each term `[a, i, j]` is c^a r^i s^j. Products keep the printed order (c, then r, then s). In `spec`, `group` is the presentation written with `x` for the direct product (`C3xD7`).
- A matrix over the group algebra is written in brackets, entries separated by `,` and rows by `;`, for example `[a, b; c, d]` or `[a, b]`. A single entry is written without brackets.
- The LP compiler applies the required transpose and group-element inversion to B internally; `A` and `B` are the inputs as printed.
- HGP: G is `--`, A = H_1 and B = H_2 are the two classical parity-check matrices, and `I_6+S_6` is the 6×6 identity plus the cyclic shift matrix. A seed matrix of another form is written row by row, `H_2=[10010000; 01001000; ...]`, each row a string of binary digits. In `spec`, `seeds.H1` and `seeds.H2` are these matrices.

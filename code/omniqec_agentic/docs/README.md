# OmniQEC design documents

This directory holds the design documents (01–03a) and the implementation snapshot (04) of the framework, written during development (mostly in Chinese). Read them in numerical order, or start with [04](04_architecture.md), which describes what the code actually runs.

| No. | Document | Scope |
|---|---|---|
| 01 | [Code families](01_code_family_design.md) | The four search grammars (bb / hgp / lp_abelian / lp_nonabelian), seed grammars, group catalogue and element encoding, deformation module |
| 02 | [Syndrome extraction and noise](02_sec_design.md) | Left-Right circuit synthesis, ranking and audits, circuit-distance evidence, noise-profile architecture, decoder choice |
| 03 | [Workflow and learning layer](03_workflow_design.md) | The campaign workflow (orchestrator and mixed-family search), LangGraph layer, learning layer, racing, qualification |
| 03a | [Agent prompts](03a_agent_prompts.md) | Prompts of the two LLM roles and their assembly rules |
| 04 | [Architecture and data flow](04_architecture.md) | As-built snapshot: layers, LLM roles, data flow of one iteration, measurement protocol, artefacts, frozen protocol and resume |

The design documents are frozen once accepted; later deviations are recorded only in appended implementation-revision sections. Order of authority: code > 04 > design documents.

The settings used in the paper are described in [final_protocol_20261106.md](final_protocol_20261106.md) and the `paper_alignment_*.md` documents. Mentions of deployment scripts, site manuals and run records that are not part of this repository are historical notes only.

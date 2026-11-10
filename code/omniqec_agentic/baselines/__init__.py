"""Search baselines without an LLM (2026-11-06, user decision).

Two complete optimizers that share the Full search's construction, verification and measurement code and replace
only the two places where the Full search uses an LLM: who proposes the codes (a fixed sampling prior,
``sampler``) and which codes of the cumulative legal pool reach the slow loop (uniform for ``random``, a Gaussian
process over a graph embedding for ``bo``, ``bo``/``selection``). The ablation figure adds the random baseline over
the four families; Fig.2 / Fig.3 add the BO baseline over BB codes only."""

"""Regenerate the S3-ablation 'default off' goldens (2026-10-23).

Run from the directory containing omniqec_agentic/ with the SAME tree state
whose behaviour must be preserved (the exp/fig3-cnot-v2 tag):

    python omniqec_agentic/tests/golden/s3_ablation/_regen_s3_goldens.py

tests/test_s3_ablation.py asserts that, with the ablation switch off, the
generator prompt, the orchestrator context blocks and the protocol snapshot
are byte-identical to these files. Not a test module (no test_ prefix)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from omniqec_agentic.tests.s3_ablation_fixtures import (  # noqa: E402
    orchestrator_blocks_text,
    prompt_text,
    snapshot_dict,
)


def main() -> None:
    for task in ("memory", "cnot_dense"):
        for mode in ("both", "fast", "none"):
            (HERE / f"prompt_{task}_{mode}.txt").write_text(prompt_text(task, mode), encoding="utf-8")
        (HERE / f"snapshot_{task}.json").write_text(
            json.dumps(snapshot_dict(task), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    with tempfile.TemporaryDirectory() as tmp:
        (HERE / "orchestrator_blocks.txt").write_text(orchestrator_blocks_text(Path(tmp)), encoding="utf-8")
    print("goldens written to", HERE)


if __name__ == "__main__":
    main()

"""Agentic Orchestrator tool loop (03 §3.2, 03a §3).

Hand-written loop over the completion-only llm/ client: the model replies
with exactly one JSON object per message — {"tool": name, "arguments": {...}}
to call a registry tool, or {"submit_plan": {...}} to close the turn. The
TurnPlan is schema-validated with citation checking; one structured repair
round, then the FallbackPolicy closes the turn (03a §3.3).

Traceability (03a R2/R5): every context block is a tagged <source> element
with an origin and content hash; the block registry, every raw LLM response
(written to disk BEFORE parsing), every tool call, and the final plan land in
the turn's transcript directory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from omniqec_agentic import ranking
from omniqec_agentic.config import LLMConfig

from .manual import render_dynamic, render_static
from .plan import TurnPlan, validate_turn_plan
from .policies import FallbackPolicy
from .prompts import load_prompt
from .registry import ToolRuntime
from omniqec_agentic.tools.context import ToolContext

# S3 ablation no_slow (2026-10-23): the one block that differs from a formal
# turn's context. Facts about the protocol, no advice about what to propose.
NO_SLOW_NOTICE = (
    "ABLATION NOTICE (protocol variant: ablation=no_slow, appendix S3 'OmniQEC w/o "
    "slow'). This campaign runs NO circuit-level (slow-loop) measurement during the "
    "search. Consequences: S-4 and S-5 are empty by design on every turn, there is no "
    "runtime hall of fame, acquisition_score and named promotions have no effect "
    "(no slow queue exists), and the generator sees the structural layer only. The "
    "structural retention record (S-3, S-9) is the only evidence of this campaign; "
    "after the last turn the frozen final stage measures the structural front's slot "
    "order once. Plan the next generation from S-3/S-9 and the ledger; do not treat "
    "the empty measurement blocks as 'not measured yet'."
)

_JSON_RE = re.compile(r"\{.*\}", re.S)

# Fields that only ever appear in a submitted plan. Used to recognise a plan
# that the model wrote without its {"submit_plan": ...} envelope.
_PLAN_FIELDS = frozenset({"observations", "program_count", "promotions", "rationale"})


def _state_budget(ctx: "ToolContext") -> int | None:
    """The run's TOTAL qubit budget from the light state (spec.qubit_budget)."""

    spec = (ctx.state or {}).get("spec") or {}
    try:
        value = int(spec.get("qubit_budget") or 0)
    except (TypeError, ValueError):
        return None
    return value or None


def _coverage_payload(ctx: "ToolContext") -> dict[str, Any]:
    """Structural coverage for S-9, trimmed to what fits a context block.

    v3.6 (docs/06 §4.3): regions carry LP shapes and the utilisation band at
    this budget; ``catalog_gaps`` names the (branch x shape) pairs never
    proposed at this budget with per-region best k / d_search; ``fact_lines``
    are the same numbers rendered as neutral sentences (no goal verb) so the
    strategy brief and this block say exactly the same thing."""

    from omniqec_agentic.memory.coverage import catalog_gaps, coverage_fact_lines, coverage_report

    try:
        n_budget = _state_budget(ctx)
        payload = coverage_report(ctx.archive_rows, ctx.slow_rows, limit=16, n_budget=n_budget)
        gaps = catalog_gaps(ctx.archive_rows, n_budget)
        gaps["regions"] = gaps["regions"][:16]
        gaps["never_proposed_branch_shapes"] = gaps["never_proposed_branch_shapes"][:24]
        payload["lp_nonabelian_catalog"] = gaps
        payload["fact_lines"] = coverage_fact_lines(ctx.archive_rows, n_budget, limit=12)
        return payload
    except Exception as exc:  # a broken instrument must not kill the turn
        return {"error": f"coverage unavailable: {exc}"}


def _remaining_capacity(state: dict[str, Any]) -> dict[str, Any]:
    """What is left of the run's evaluation capacity when this turn is planned:
    turns (this one included), the slow-loop evaluation slots of one turn and
    how many of them a plan may name (Table I "remaining evaluation capacity")."""

    protocol = state.get("protocol") or {}
    retention = protocol.get("structural_retention") or {}
    try:
        max_turns = int(protocol.get("max_turns") or 0)
        used = int((state.get("budget") or {}).get("turns_used") or 0)
        return {
            "turns_including_this_one": max(0, max_turns - used),
            "slow_evaluation_slots_per_turn": int(protocol.get("promote_top") or 0),
            "nominable_runtime_slots_per_turn": int(retention.get("runtime_slow_slots") or 0),
        }
    except (TypeError, ValueError):
        return {}


def _reflection_payload(ctx: "ToolContext") -> str:
    """S-10 body: decayed reflection window (see orchestrator/reflector.py)."""

    from omniqec_agentic.orchestrator.reflector import read_reflections, render_reflections

    try:
        return render_reflections(read_reflections(ctx.manual_dir))
    except Exception as exc:  # a broken instrument must not kill the turn
        return f"(reflections unavailable: {exc})"


def _accepts_segments(client: Any) -> bool:
    """Does this client take the segmented form of the conversation?"""

    import inspect

    try:
        return "user_segments" in inspect.signature(client.complete).parameters
    except (TypeError, ValueError):
        return False


def _tool_requests(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Normalise both envelopes to one list of (name, arguments).

    `{"tool": n, "arguments": {...}}` is the original single call and is
    unchanged; `{"tools": [ ... ]}` is the batch form. Malformed entries are
    dropped here so a half-valid batch still executes its valid part rather
    than costing a repair round.
    """

    if "tool" in payload:
        return [(str(payload.get("tool")), dict(payload.get("arguments") or {}))]
    requests: list[tuple[str, dict[str, Any]]] = []
    for item in payload.get("tools") or []:
        if isinstance(item, dict) and item.get("tool"):
            requests.append((str(item["tool"]), dict(item.get("arguments") or {})))
    return requests


def _block(source_id: str, kind: str, origin: str, content: str) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256(content.encode()).hexdigest()
    text = (
        f'<source id="{source_id}" kind="{kind}" origin="{origin}" hash="sha256:{digest[:16]}">\n'
        f"{content}\n</source>"
    )
    return text, {"id": source_id, "kind": kind, "origin": origin, "hash": digest[:16]}


def _compact_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], limit: int) -> str:
    lines = []
    for row in rows[:limit]:
        parts = []
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            if key == "code_hash":
                value = str(value)[:8]
            parts.append(f"{key}={value}")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines) if lines else "  (none)"


class AgenticPolicy:
    """LLM policy agent behind the same Policy protocol as FallbackPolicy."""

    def __init__(
        self,
        *,
        provider: str,
        llm_config: LLMConfig | None = None,
        generations: int = 1,
        rng_seeds_per_program: int = 1,
        allow_fallback: bool = True,
        fixed_program_count: bool = False,
        format_repair_budget: int = 4,
    ) -> None:
        self.provider = str(provider)
        self.llm_config = llm_config or LLMConfig()
        self.generations = int(generations)
        self.rng_seeds_per_program = int(rng_seeds_per_program)
        self.allow_fallback = bool(allow_fallback)
        self.fixed_program_count = bool(fixed_program_count)
        self.format_repair_budget = max(1, int(format_repair_budget))
        self._fallback = FallbackPolicy(
            provider=provider, generations=generations,
            rng_seeds_per_program=rng_seeds_per_program,
        )

    # ---- context assembly (03a §3.2) -----------------------------------
    def _assemble_blocks(
        self, state: dict[str, Any], ctx: ToolContext
    ) -> tuple[str, list[dict[str, Any]]]:
        from .registry import render_tool_docs

        blocks: list[str] = []
        registry_rows: list[dict[str, Any]] = []
        manual_index = (ctx.manual_dir / "static" / "INDEX.md")
        insights = (ctx.manual_dir / "dynamic" / "insights.md")
        # S-1 carries the FULL tool catalog inline (03a §3.2: 工具目录 +
        # 模块摘要) — the model must never need a discovery call to learn a
        # tool's signature.
        index_text = manual_index.read_text(encoding="utf-8") if manual_index.exists() else "(empty)"
        pairs = [
            ("S-1", "manual_static_index", "manual/static/INDEX.md+tools.md",
             index_text + "\n\n" + render_tool_docs()),
            ("S-2", "manual_dynamic", "manual/dynamic/insights.md",
             insights.read_text(encoding="utf-8") if insights.exists() else "(none yet)"),
            # S-3 is the STRUCTURAL ledger (v3.2 §5.2): the (k, d_search)
            # proxy-Pareto elites with their rank and crowding coordinates.
            # The old Q-ordered fast queue made the policy see one ranking
            # only, so even neutral wording steered it back to that axis.
            ("S-3", "structural_frontier", "state.structural_proxy_elites",
             _compact_rows(state.get("structural_proxy_elites") or [],
                           ("code_hash", "family", "n", "n_total", "k", "d_search",
                            # v3.7.1: the front's third axis. It decided
                            # retention from v3.7 on and was never shown here.
                            "max_check_weight",
                            "pareto_rank", "crowding_boundary"), 16)),
            ("S-4", "digest_slow", "last_cycle/slow_rankings",
             _compact_rows([r for r in ctx.slow_rows if not r.get("is_baseline")],
                           ("code_hash", "family", "max_check_weight",
                            "per_logical_per_round",
                            "per_logical_block_ci", "d_circ_ext", "degraded_ext",
                            "stopped_by", "status"), 12)),
            ("S-5", "hof_baseline", "state.hof_summary+baseline_summary",
             json.dumps({"hall_of_fame": state.get("hof_summary") or [],
                         "baseline": state.get("baseline_summary") or {}},
                        ensure_ascii=False, indent=1, sort_keys=True, default=str)),
            ("S-6", "budget", "state.budget",
             json.dumps({"used": state.get("budget") or {},
                         "caps": {"max_turns": (state.get("protocol") or {}).get("max_turns"),
                                  "tool_budget": (state.get("protocol") or {}).get("tool_budget")},
                         # Paper Table I ("search guidance: ... remaining
                         # evaluation capacity"), 2026-11-10 alignment.
                         "remaining": _remaining_capacity(state)},
                        sort_keys=True)),
            ("S-7", "bandit", "bandit.suggestion()",
             json.dumps(ctx.bandit.suggestion(), sort_keys=True)),
            ("S-8", "last_plan_outcome", "state.plan+last_cycle.status",
             json.dumps({"last_plan": state.get("plan"),
                         "last_cycle_status": (state.get("last_cycle") or {}).get("status")},
                        ensure_ascii=False, sort_keys=True, default=str)),
            # S-9 answers the one question the ranked blocks above cannot: what
            # has NEVER been tried. Regions with zero proposals leave no row in
            # S-3/S-4, so without this block an unexplored corner of the grammar
            # is indistinguishable from one that was explored and lost.
            ("S-9", "structural_coverage", "coverage_report(archive+slow)",
             json.dumps(_coverage_payload(ctx), ensure_ascii=False,
                        sort_keys=True, indent=1, default=str)),
            # S-10 is advice, not evidence. It carries what recent turns
            # concluded in prose, newest in full and older ones down to one
            # line, so a lesson has a limited time to either become a ledger
            # entry or be forgotten. Nothing here may be cited as a result.
            ("S-10", "recent_reflections", "manual/dynamic/reflections.jsonl",
             _reflection_payload(ctx)),
        ]
        # S-0 (S3 ablation no_slow, 2026-10-23): the planner is told, once
        # per turn, that this campaign measures nothing during the search --
        # otherwise it reads the empty S-4/S-5 as "not measured yet" and keeps
        # naming promotions that no slow queue will ever take. Absent from
        # formal runs (no key in the snapshot).
        if str((state.get("protocol") or {}).get("ablation") or "full") == "no_slow":
            pairs.insert(0, ("S-0", "ablation_notice", "protocol.ablation", NO_SLOW_NOTICE))
        # S-11 (experiment variant, 2026-09-09): frozen background codes from
        # the protocol snapshot; absent from formal runs.
        reference = (state.get("protocol") or {}).get("reference_codes")
        if isinstance(reference, dict) and reference.get("codes"):
            from .reference_codes import ORCHESTRATOR_SOURCE_ID, render_reference_codes
            pairs.append((ORCHESTRATOR_SOURCE_ID, "reference_codes", "protocol.reference_codes",
                          render_reference_codes(reference)))
        for source_id, kind, origin, content in pairs:
            text, row = _block(source_id, kind, origin, content)
            blocks.append(text)
            registry_rows.append(row)
        return "\n\n".join(blocks), registry_rows

    # ---- llm plumbing (03a R5) -----------------------------------------
    def _client(self):
        from omniqec_agentic.llm.factory import create_completion_client

        return create_completion_client(self.provider, self.llm_config)

    def _record_call(
        self, transcript_dir: Path, *, turn: int, index: int,
        prompt_hash: str, user_text: str, response: str, client_info: dict[str, Any],
    ) -> None:
        raw_path = transcript_dir / f"llm_response_{index:02d}.txt"
        raw_path.write_text(response, encoding="utf-8")  # disk BEFORE parsing
        record = {
            "turn": int(turn), "role": "orchestrator", "call_index": index,
            "prompt_id": f"orchestrator_system:{prompt_hash}",
            "model": client_info, "input_hash": hashlib.sha256(user_text.encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            "raw_response_path": str(raw_path),
        }
        with (transcript_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Return the first complete JSON object in a reply.

        Thinking models routinely emit several objects back to back, wrap the
        call in markdown or XML, or continue the transcript with invented tool
        results. Every one of those still opens with the object the model
        actually meant, so scan for the first balanced {...} rather than
        demanding that the whole reply be a single object.
        """
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text, start)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    # ---- the loop -------------------------------------------------------
    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        protocol = state.get("protocol") or {}
        run_dir = Path(state["run_dir"])
        turn = int(state.get("turn") or 0)
        manual_dir = run_dir / "manual"
        transcript_dir = run_dir / "turns" / f"turn_{turn:04d}" / "orchestrator"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        ctx = ToolContext(run_dir=run_dir, manual_dir=manual_dir, state=dict(state), turn=turn)
        render_static(manual_dir)
        render_dynamic(manual_dir, ledger=ctx.ledger, bandit=ctx.bandit,
                       digest=(state.get("last_cycle") or {}).get("artifact_paths"))

        system_text, prompt_hash = load_prompt("orchestrator_system")
        tool_budget = int(protocol.get("tool_budget") or 32)
        system_text = system_text.replace("{tool_budget}", str(tool_budget))
        runtime = ToolRuntime(context=ctx, tool_budget=tool_budget)

        context_text, block_registry = self._assemble_blocks(state, ctx)
        (transcript_dir / "context_blocks.json").write_text(
            json.dumps(block_registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_ids = {row["id"] for row in block_registry}
        known_hashes = {
            str(r.get("code_hash")) for r in ctx.archive_rows + ctx.slow_rows if r.get("code_hash")
        }
        # v3.2 §4.4/§5.2: SEC hook degradation is a diagnostic label, not a
        # quarantine. Its cost is already priced into the measured plpr, and
        # excluding degraded candidates deleted structurally strong codes from
        # the search on the strength of a runtime symptom. Only a structural
        # impossibility stays a hard exclusion: no sampled schedule avoids a
        # single-fault logical, so the code cannot be measured at all.
        quarantined = {
            str(r.get("code_hash"))
            for r in ctx.slow_rows
            if r.get("schedule_catastrophic")
        }
        # Deterministic acquisition baseline (03 §4.2, v3.2 §4.3): the archive
        # queue is archive_order_key-ordered, so this is the baseline for the
        # runtime slots (2026-09-10: two) — the structural slots are
        # machine-produced and are not TurnPlan promotions at all.
        # ONE definition, shared with the acquisition_score tool the model
        # consults (ranking.runtime_slot_baseline): what the tool reports as
        # baseline_order is exactly what the validator checks against.
        acquisition_order = ranking.runtime_slot_baseline(ctx.archive_rows, ctx.slow_rows)

        conversation: list[str] = [
            context_text,
            f"\nTurn {turn}. You have {tool_budget} tool calls; a typical turn "
            "needs 2-6 (the full tool catalog with signatures is already in "
            "S-1 — no discovery calls needed). Investigate as needed, then "
            "close the turn with {\"submit_plan\": {...}}.",
        ]
        try:
            client = self._client()
            client_info = dict(client.info())
        except Exception as exc:
            self._note(transcript_dir, f"llm_client_unavailable: {exc}")
            return self._fallback_plan(state, transcript_dir, reason=f"client: {exc}")

        # Formatting repairs are a presentation channel, not a scientific one:
        # the model still has to produce a fully valid plan itself. One repair
        # per turn made a single slip fatal under --strict-turns, so allow a
        # small budget and let a clean call restore one chance.
        repairs_left = int(self.format_repair_budget)
        for call_index in range(tool_budget + 2):
            user_text = "\n\n".join(conversation)
            try:
                # The conversation only ever grows by appending, so handing the
                # client the pieces (rather than the joined string) lets a
                # provider that caches at block boundaries read everything but
                # the newest segment from cache. Clients without the parameter
                # get the joined text exactly as before.
                if _accepts_segments(client):
                    response = client.complete(
                        system=system_text, user=user_text, user_segments=list(conversation)
                    )
                else:
                    response = client.complete(system=system_text, user=user_text)
            except Exception as exc:
                self._note(transcript_dir, f"llm_call_failed: {exc}")
                return self._fallback_plan(state, transcript_dir, reason=f"llm: {exc}")
            try:
                client_info = dict(client.info())
            except Exception:
                pass
            self._record_call(
                transcript_dir, turn=turn, index=call_index,
                prompt_hash=prompt_hash, user_text=user_text, response=response,
                client_info=client_info,
            )
            payload = self._parse_json(response)
            if payload is None:
                if repairs_left <= 0:
                    return self._fallback_plan(
                        state, transcript_dir, reason="unparseable beyond repair budget"
                    )
                repairs_left -= 1
                conversation.append(
                    "VALIDATION ERROR: reply was not a single JSON object. Reply with "
                    'exactly one JSON object: {"tool": ...} or {"submit_plan": ...}.'
                )
                continue
            if "submit_plan" not in payload and "tool" not in payload and "tools" not in payload:
                # The plan body without its envelope is unambiguous; validate it
                # as a submission rather than burning a repair on the wrapper.
                if _PLAN_FIELDS & set(payload):
                    payload = {"submit_plan": payload}
            if "submit_plan" in payload:
                ledger_ids = {
                    e.display_id for e in ctx.ledger.entries.values()
                } | {f"H-{e.id}" for e in ctx.ledger.entries.values()} | {
                    f"I-{e.id}" for e in ctx.ledger.entries.values()
                }
                plan_obj, errors = validate_turn_plan(
                    payload["submit_plan"],
                    source_ids=source_ids,
                    known_hashes=known_hashes,
                    ledger_ids=ledger_ids,
                    families=tuple((state.get("spec") or {}).get("families") or ()),
                    program_cap=int(protocol.get("proposal_count") or 8) * 4,
                    # At most runtime_slow_slots named candidates (2026-09-10:
                    # two); each consumes a runtime slot rather than adding one
                    # (v3.2 §4.3).
                    promote_cap=int(
                        (protocol.get("structural_retention") or {}).get("runtime_slow_slots")
                        or 1
                    ),
                    quarantined_hashes=quarantined,
                    acquisition_order=acquisition_order,
                )
                expected_program_count = int(protocol.get("proposal_count") or 8)
                if (
                    plan_obj is not None
                    and self.fixed_program_count
                    and int(plan_obj.program_count) != expected_program_count
                ):
                    errors.append(
                        "program_count: strict campaign requires exactly "
                        f"{expected_program_count} (got {plan_obj.program_count})"
                    )
                    plan_obj = None
                if plan_obj is None:
                    if repairs_left <= 0:
                        return self._fallback_plan(
                            state, transcript_dir,
                            reason=f"plan invalid beyond repair budget: {errors[:3]}",
                        )
                    repairs_left -= 1
                    conversation.append(
                        "PLAN REJECTED. Fix these and resubmit the full plan:\n- "
                        + "\n- ".join(errors)
                    )
                    continue
                return self._accept_plan(plan_obj, state, transcript_dir, runtime)
            if "tool" in payload or "tools" in payload:
                if runtime.remaining() <= 0:
                    conversation.append(
                        "TOOL BUDGET EXHAUSTED. Close the turn now with {\"submit_plan\": {...}}."
                    )
                    continue
                # One round trip may carry several independent reads. Each tool
                # still counts against the same budget and is recorded
                # identically; what is saved is the RESEND — every extra round
                # trip re-sends the whole conversation and pays for a fresh
                # preamble of reasoning (measured on opus-5: 10 round trips,
                # 127k input and 22.7k output tokens per turn). Dependent
                # lookups can still be issued one at a time; nothing here
                # forces batching.
                requests = _tool_requests(payload)
                if not requests:
                    if repairs_left <= 0:
                        return self._fallback_plan(
                            state, transcript_dir,
                            reason="empty tool request beyond repair budget",
                        )
                    repairs_left -= 1
                    conversation.append(
                        'VALIDATION ERROR: "tools" must be a non-empty list of '
                        '{"tool": name, "arguments": {...}} objects.'
                    )
                    continue
                blocks: list[str] = []
                for name, arguments in requests:
                    if runtime.remaining() <= 0:
                        blocks.append(f"TOOL SKIPPED {name}: tool budget exhausted")
                        continue
                    result = runtime.call(name, arguments)
                    blocks.append(
                        f"TOOL RESULT for {name}:\n"
                        + json.dumps(result, ensure_ascii=False, sort_keys=True,
                                     default=str)[:6000]
                    )
                conversation.append(
                    "\n\n".join(blocks)
                    + f"\n\n({runtime.remaining()} tool calls left)"
                )
                repairs_left = min(int(self.format_repair_budget), repairs_left + 1)
                continue
            if repairs_left <= 0:
                return self._fallback_plan(
                    state, transcript_dir, reason="no actionable key beyond repair budget"
                )
            repairs_left -= 1
            conversation.append(
                'VALIDATION ERROR: the object must contain "tool" or "submit_plan". '
                "Reply with exactly one JSON object and nothing else."
            )
        return self._fallback_plan(state, transcript_dir, reason="loop budget exhausted")

    # ---- plan closing ---------------------------------------------------
    def _accept_plan(
        self,
        plan_obj: TurnPlan,
        state: dict[str, Any],
        transcript_dir: Path,
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        protocol = state.get("protocol") or {}
        (transcript_dir / "turn_plan.json").write_text(
            json.dumps(plan_obj.model_dump(mode="json"), ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (transcript_dir / "tool_transcript.json").write_text(
            json.dumps(runtime.transcript, ensure_ascii=False, indent=2, sort_keys=True,
                       default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "schema": "turn_plan",
            "policy": "agentic_v1",
            "turn": int(state.get("turn") or 0),
            "provider": self.provider,
            "generations": self.generations,
            "proposal_count": int(plan_obj.program_count),
            "rng_seeds_per_program": self.rng_seeds_per_program,
            "families": list((state.get("spec") or {}).get("families") or []),
            "family_mix": plan_obj.family_mix,
            "strategy_mix": plan_obj.strategy_mix,
            "explore_temperature": float(plan_obj.explore_temperature),
            # Always the frozen cap: a plan's promotion count must not resize
            # the slow queue (v3.2 §4.3).
            "promote_top": int(protocol.get("promote_top") or 4),
            "promotions": [p.model_dump(mode="json") for p in plan_obj.promotions],
            "budget_class": "pilot",
            "directives": plan_obj.directives,
            "feedback_mode": str(protocol.get("feedback_mode") or "both"),
            "finalize": bool(plan_obj.finalize),
            "rationale": plan_obj.rationale.model_dump(mode="json"),
            "observations": [o.model_dump(mode="json") for o in plan_obj.observations],
            "tool_calls_used": runtime.calls_used,
        }

    def _fallback_plan(
        self, state: dict[str, Any], transcript_dir: Path, *, reason: str
    ) -> dict[str, Any]:
        self._note(transcript_dir, f"fallback_policy_closed_turn: {reason}")
        if not self.allow_fallback:
            raise RuntimeError(
                "strict campaign forbids fallback turns; retry this turn from "
                f"checkpoint ({reason})"
            )
        plan = self._fallback.plan(state)
        plan["policy"] = "agentic_v1_fallback"
        plan["fallback_reason"] = reason
        return plan

    @staticmethod
    def _note(transcript_dir: Path, message: str) -> None:
        with (transcript_dir / "events.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

"""REFINE-HT scenario: REFINE-CV (same-model) + explicit cross-round Hypothesis Tracker.

Targets the "correct hypothesis dropped" failure mode (recency bias) identified
in the failure-root-cause analysis: the diagnostician regenerates a single
diagnosis from scratch each round, so a correct candidate conceived early is
silently erased by re-summarisation.

The tracker differs from REFINE-HPR (which had a negative result): HPR forced
the diagnostician to choose FROM a generated DDx list, shifting the decision
onto an unstable list generator. Here the tracker is injected as ADDITIONAL
memory context — the diagnostician still reasons freely, but is reminded of
every candidate it has already conceived.

Pipeline (vs REFINE-CV):
    REFINE-CV  : summarizer -> diagnostician -> verifier -> rule tree
    REFINE-HT  : summarizer -> [memory] diagnostician -> verifier -> rule tree
                 -> [tracker.register(diagnosis, evidence, refutation)]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from eid.hypothesis_tracker import HypothesisTracker
from eid.scenarios.refine_cv_qwen import RefineCVQwenScenario

if TYPE_CHECKING:
    from eid.agents.chat_agent import AgentWrapper


class RefineHTScenario(RefineCVQwenScenario):
    """REFINE-CV (same-model verification) with a cross-round hypothesis tracker."""

    def run(self, case_input):
        # One tracker per case: it must NOT leak across cases.
        self._tracker = HypothesisTracker()
        return super().run(case_input)

    # ------------------------------------------------------------------ #
    # Pipeline override: identical to REFINE-CV except the tracker hooks
    # ------------------------------------------------------------------ #
    def _run_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        current_turn: int,
        prev_diagnosis: str | None,
    ) -> tuple[str, str | None, str]:
        tracker = self._tracker
        tracker.begin_round()

        # 1) Summarizer (unchanged)
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": summarizer.get_last_duration(),
        })
        summary = self.extract_summary(s_response)

        # 2) Diagnostician — inject the accumulated hypothesis memory as
        #    ADDITIONAL context (never a constraint).
        d_instruction = self.prompts.get_diagnostician_instruction(summary)
        memory = tracker.render()
        if memory:
            d_instruction = d_instruction.replace(
                "Diagnostician:", memory + "\n\nDiagnostician:"
            )
        d_response = diagnostician.step(d_instruction)
        trace.append({
            "role_id": "diagnostician",
            "content": d_response,
            "duration": diagnostician.get_last_duration(),
        })

        # 2b) Register the current diagnosis + its cited evidence.
        _, _current_dx = self.extract_action(d_response)
        tracker.register(_current_dx, supporting=self._extract_evidence(d_response))

        # 3) Cross-model verification (unchanged)
        verification = self._verify(
            summary, d_response, prev_diagnosis, current_diagnosis=_current_dx
        )
        trace.append({
            "role_id": "verifier",
            "content": verification,
            "duration": None,
        })

        # 3b) Register the verifier's refutation: counter-evidence argues
        #     against the current diagnosis; the alternative becomes a new
        #     tracked candidate.
        ref = (verification.get("refutation", {}) or {})
        counter = (ref.get("counter_evidence") or "").strip()
        alt = (ref.get("alternative") or "").strip()
        if counter:
            tracker.register(_current_dx, refuting=[counter])
        if alt:
            tracker.register(alt)

        # 4) Deterministic rule tree (unchanged, except re-register on revision)
        bad = self._bad_claims(verification) if self.ENABLE_CLAIM_CHECK else []
        if bad:
            correction = "\n".join(
                f"- '{c.get('statement', '')}' -> {c.get('verdict', '')}"
                for c in bad
            )
            r_instruction = self.prompts.get_diagnostician_cv_revision_instruction(
                summary, correction
            )
            d_response = diagnostician.step(r_instruction)
            trace.append({
                "role_id": "diagnostician",
                "content": d_response,
                "duration": diagnostician.get_last_duration(),
            })
            _, _current_dx = self.extract_action(d_response)
            tracker.register(_current_dx, supporting=self._extract_evidence(d_response))
            verification = self._verify(
                summary, d_response, prev_diagnosis, current_diagnosis=_current_dx
            )
            trace.append({
                "role_id": "verifier",
                "content": verification,
                "duration": None,
            })
            bad = self._bad_claims(verification)

        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response

        # 5) Decide next action (unchanged)
        refutation = (verification.get("refutation", {}) or {}) if self.ENABLE_REFUTATION else {}
        gaps = (verification.get("gaps", []) or []) if self.ENABLE_GAP_CHECK else []
        decisive = bool(verification.get("decisive", True))

        anchored = (
            self.ENABLE_REFUTATION
            and prev_diagnosis is not None
            and self._same_diagnosis(prev_diagnosis, answer)
        )

        if anchored and current_turn < self.max_turns - 1:
            counter = refutation.get("counter_evidence", "") or ""
            alt = refutation.get("alternative", "") or ""
            feedback = (
                f"The diagnosis '{answer}' is unchanged from the previous round. "
                f"Counter-evidence to verify: {counter}. "
                f"Also consider this not-yet-excluded alternative: {alt}."
            )
            self._gather(doctor, patient, reporter, trace, feedback)
            # persist tracker state into the trace for debugging
            trace.append({"role_id": "hypothesis_tracker", "content": tracker.to_dict(), "duration": None})
            return answer, answer, "GATHER"

        if gaps and not decisive and current_turn < self.max_turns - 1:
            top = gaps[0]
            feedback = (
                f"Missing discriminating information: {top.get('info', '')} "
                f"(discriminates: {top.get('discriminates', '')})."
            )
            self._gather(doctor, patient, reporter, trace, feedback)
            trace.append({"role_id": "hypothesis_tracker", "content": tracker.to_dict(), "duration": None})
            return answer, answer, "GATHER"

        # No hallucination left, no critical gap, no anchor → accept.
        trace.append({"role_id": "hypothesis_tracker", "content": tracker.to_dict(), "duration": None})
        return answer, answer, "PASS"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_evidence(d_response: str) -> list[str]:
        """Extract the diagnostician's [EVIDENCE] statements as a flat list."""
        m = re.search(
            r"\[EVIDENCE\]\s*([\s\S]*?)(?=\[GAPS\]|\[DIAGNOSIS\]|$)",
            d_response or "",
        )
        if not m:
            return []
        lines = [ln.strip(" -*\u2022").strip() for ln in m.group(1).splitlines()]
        return [ln for ln in lines if ln and len(ln) > 3][:5]

"""DiagnosticGate: a method-agnostic, pluggable verification gate.

This is the "plugin" extracted from the REFINE-CV experiments. It is *not*
tied to any single orchestration method (REFINE / MEDDxAgent / ClinicalAgents):
it only needs two inputs that every method can produce at the moment it is
about to accept a diagnosis:

1. ``evidence_state``  - all clinical information gathered so far (REFINE's
   summary, MEDDxAgent's patient profile, ClinicalAgents' EVIDENCE block, ...);
2. ``diagnosis``       - the candidate diagnosis the host method just produced;
3. ``prev_diagnosis``  - the diagnosis produced in the previous gate round
   (used for the deterministic anchoring trigger).

The gate runs the three audits validated in REFINE-CV, then returns one of four
deterministic decisions via a rule tree (NOT a free-form LLM verdict):

- PASS         : accept the diagnosis;
- REDO         : evidence is broken (contradicted / hallucinated / undetermined)
                 -> re-diagnose without the flagged findings;
- BREAK_ANCHOR : diagnosis unchanged across rounds -> force one counter-evidence
                 plus one unexcluded alternative;
- GATHER_GAP   : a discriminating piece of information is still missing.

Design constraints learned from the earlier ablations (do NOT regress these):
- the verifier runs on the SAME model as the host doctor (cross-model semantic
  misalignment collapsed to 40%);
- the three audits must run together (any single one alone is no better than
  the baseline);
- the decision is a rule tree, not an LLM verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from llm_client import get_llm, chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import get_llm, chat as _llm_chat

# Verdicts that mean the diagnosis is NOT trustworthy (evidence is broken).
_BAD_VERDICTS = {"CONTRADICTED", "HALLUCINATED", "UNDETERMINED"}


@dataclass
class GateDecision:
    """Result of one gate evaluation."""

    action: str  # "PASS" | "REDO" | "BREAK_ANCHOR" | "GATHER_GAP"
    feedback: str = ""  # feedback to feed back to the host method (empty on PASS)
    verification: dict[str, Any] = field(default_factory=dict)  # raw verifier JSON


class DiagnosticGate:
    """Pluggable, method-agnostic verification gate."""

    def __init__(self, prompts: Any, provider: str = "qwen", model: str | None = None) -> None:
        """Initialize the gate.

        Args:
            prompts: A PromptManager exposing get_verifier_gate_{system,user}_prompt.
            provider: LLM provider for the verifier. Defaults to "qwen" so the
                verifier shares the host doctor's model (same-model verification).
            model: Optional explicit model name; None = provider default.
        """
        self.prompts = prompts
        self._client, self._model = get_llm(provider)
        if model:
            self._model = model

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def _verify(self, evidence_state: str, diagnosis: str, prev_diagnosis: str | None) -> dict:
        """Call the same-model verifier and return its parsed JSON."""
        sys_prompt = self.prompts.get_verifier_gate_system_prompt()
        user_prompt = self.prompts.get_verifier_gate_user_prompt(
            evidence_state, diagnosis, prev_diagnosis
        )
        try:
            resp = _llm_chat(
                self._client,
                self._model,
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                json_mode=True,
            )
        except Exception:  # pragma: no cover - fail open on verifier errors
            resp = None
        if not isinstance(resp, dict):
            return {"claim_checks": [], "gaps": [], "refutation": {"triggered": False}}
        return resp

    @staticmethod
    def _bad_claims(verification: dict) -> list[dict]:
        """Return claim-check entries whose verdict is untrustworthy."""
        return [
            c
            for c in verification.get("claim_checks", [])
            if str(c.get("verdict", "")).upper() in _BAD_VERDICTS
        ]

    # ------------------------------------------------------------------ #
    # Rule tree
    # ------------------------------------------------------------------ #
    def decide(
        self,
        evidence_state: str,
        diagnosis: str,
        prev_diagnosis: str | None,
        current_turn: int,
        max_turns: int,
    ) -> GateDecision:
        """Run verification and the deterministic rule tree.

        Returns a GateDecision whose action is one of PASS / REDO /
        BREAK_ANCHOR / GATHER_GAP. Priority: REDO > BREAK_ANCHOR > GATHER_GAP
        > PASS (matches the validated REFINE-CV order).
        """
        verification = self._verify(evidence_state, diagnosis, prev_diagnosis)

        bad = self._bad_claims(verification)
        if bad:
            correction = "\n".join(
                f"- '{c.get('statement', '')}' -> {c.get('verdict', '')}" for c in bad
            )
            feedback = (
                "Your diagnosis relies on evidence that failed independent verification:\n"
                f"{correction}\n"
                "Re-examine the evidence state and re-diagnose WITHOUT the flagged findings."
            )
            return GateDecision("REDO", feedback, verification)

        refutation = verification.get("refutation", {}) or {}
        gaps = verification.get("gaps", []) or []

        if refutation.get("triggered") and current_turn < max_turns - 1:
            feedback = (
                f"The diagnosis '{diagnosis}' is unchanged from the previous round. "
                f"Counter-evidence to verify: {refutation.get('counter_evidence', '')}. "
                f"Also consider this not-yet-excluded alternative: {refutation.get('alternative', '')}."
            )
            return GateDecision("BREAK_ANCHOR", feedback, verification)

        if gaps and current_turn < max_turns - 1:
            top = gaps[0]
            feedback = (
                f"Missing discriminating information: {top.get('info', '')} "
                f"(discriminates: {top.get('discriminates', '')})."
            )
            return GateDecision("GATHER_GAP", feedback, verification)

        return GateDecision("PASS", "", verification)

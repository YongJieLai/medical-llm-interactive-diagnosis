"""Explicit Hypothesis Tracker (cross-round DDx memory) for interactive diagnosis.

Motivation (from the failure-root-cause analysis): the REFINE / REFINE-CV
pipeline has NO explicit hypothesis state. The doctor reasons in free text and
the diagnostician regenerates a single diagnosis from scratch each round, so a
correct hypothesis conceived in an early round is silently "washed away" by the
next round's re-summarisation (recency bias) — the "correct hypothesis dropped"
failure mode (~24.5% of errors).

This module provides a lightweight, zero-training, cross-round state that:

1. ACCUMULATES every candidate diagnosis the diagnostician has considered
   across rounds (no more dropping-by-forgetting);
2. TRACKS supporting / refuting evidence per candidate;
3. RENDERS this state back to the diagnostician as ADDITIONAL context — NOT as
   a constraint — so the model keeps its free reasoning while being reminded of
   candidates it has already conceived.

Design note: this deliberately differs from REFINE-HPR, which inserted a
differential LIST the diagnostician was forced to choose FROM. HPR's negative
result showed that a constraint-list shifts the decision onto an unstable list
generator and causes regressions. The tracker here is a MEMORY augmentation,
not a decision constraint.

Lifecycle: one instance per case. `begin_round()` / `register()` are called every
verification round; `render()` is called just before the diagnostician runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def normalize_name(name: str) -> str:
    """Normalise a diagnosis name for fuzzy dedup (same rule as REFINE-CV)."""
    s = re.sub(r"['\u2019]s\b", " ", name or "")      # drop possessive 's
    s = re.sub(r"\([^)]*\)", " ", s)                  # drop parenthetical aliases
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())          # punctuation -> space
    return " ".join(s.split())


@dataclass
class TrackedHypothesis:
    name: str
    rounds_seen: list[int] = field(default_factory=list)
    supporting: list[str] = field(default_factory=list)
    refuting: list[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str = ""


class HypothesisTracker:
    """Accumulate and re-present diagnostic hypotheses across verification rounds."""

    def __init__(
        self,
        max_candidates: int = 8,
        max_evidence_per_candidate: int = 3,
    ) -> None:
        self.max_candidates = max_candidates
        self.max_evidence_per_candidate = max_evidence_per_candidate
        self._hypotheses: dict[str, TrackedHypothesis] = {}
        self.round = 0

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def begin_round(self) -> None:
        """Advance the round counter (called once per verification pipeline)."""
        self.round += 1

    def register(
        self,
        name: str,
        supporting: list[str] | None = None,
        refuting: list[str] | None = None,
        excluded: bool = False,
        exclusion_reason: str = "",
    ) -> None:
        """Merge a candidate (and its evidence) into the tracked set.

        Idempotent per round: calling twice in the same round with the same name
        only records the round once, while still accumulating new evidence.
        """
        name = (name or "").strip()
        if not name:
            return
        key = normalize_name(name)
        h = self._hypotheses.get(key)
        if h is None:
            if len(self._hypotheses) >= self.max_candidates:
                self._evict_lru()
            h = TrackedHypothesis(name=name)
            self._hypotheses[key] = h
        if self.round not in h.rounds_seen:
            h.rounds_seen.append(self.round)
        for s in supporting or []:
            s = (s or "").strip()
            if s and s not in h.supporting:
                h.supporting.append(s)
        for r in refuting or []:
            r = (r or "").strip()
            if r and r not in h.refuting:
                h.refuting.append(r)
        if excluded:
            h.excluded = True
            h.exclusion_reason = exclusion_reason or h.exclusion_reason

    def _evict_lru(self) -> None:
        """Drop the candidate seen longest ago when the set overflows."""
        oldest_key = min(
            self._hypotheses,
            key=lambda k: max(self._hypotheses[k].rounds_seen, default=0),
        )
        del self._hypotheses[oldest_key]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """Render the tracked state as a memory aid block for the diagnostician."""
        if not self._hypotheses:
            return ""
        active = [h for h in self._hypotheses.values() if not h.excluded]
        excluded = [h for h in self._hypotheses.values() if h.excluded]

        lines = [
            "### Hypotheses you have considered so far (memory aid — NOT a constraint) ###",
            "Re-evaluate freely; do NOT let re-summarisation erase a previously plausible candidate.",
        ]
        if active:
            lines.append("Still viable (not yet ruled out):")
            for h in active:
                lines.append(self._fmt(h))
        if excluded:
            lines.append("Previously considered but ruled out:")
            for h in excluded:
                lines.append(self._fmt(h))
        return "\n".join(lines)

    def _fmt(self, h: TrackedHypothesis) -> str:
        rounds = ",".join(str(r) for r in sorted(h.rounds_seen))
        sup = self._clip(h.supporting)
        ref = self._clip(h.refuting)
        parts = [f"- {h.name} (rounds {rounds})"]
        if sup:
            parts.append(f"    supporting: {'; '.join(sup)}")
        if ref:
            parts.append(f"    against: {'; '.join(ref)}")
        if h.excluded and h.exclusion_reason:
            parts.append(f"    excluded because: {h.exclusion_reason}")
        return "\n".join(parts)

    def _clip(self, items: list[str]) -> list[str]:
        return items[: self.max_evidence_per_candidate]

    # ------------------------------------------------------------------ #
    # Serialisation (for trace / debugging)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "hypotheses": [
                {
                    "name": h.name,
                    "rounds_seen": list(h.rounds_seen),
                    "supporting": list(h.supporting),
                    "refuting": list(h.refuting),
                    "excluded": h.excluded,
                    "exclusion_reason": h.exclusion_reason,
                }
                for h in self._hypotheses.values()
            ],
        }

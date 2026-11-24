"""REFINE-SC ablation: self-consistency (same-model multi-sampling) voting.

Extends REFINE with a self-consistency diagnostician: the diagnosis is sampled
N times (same Qwen model, temperature > 0) and the majority-voted diagnosis is
returned. This is the "same-model multi-sampling" counterpart to REFINE-CV's
cross-model verification — together they isolate whether cross-model decoupling
beats simply sampling the same weak model multiple times.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from eid.scenarios.refine import RefineScenario

if TYPE_CHECKING:
    from eid.config import ModelConfig

try:
    from llm_client import get_llm, chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import get_llm, chat as _llm_chat


def _norm(s: str) -> str:
    """Lowercase + strip non-alphanumerics for stable voting."""
    s = (s or "").lower().replace("'s", "")
    return re.sub(r"[^a-z0-9]", "", s)


class RefineSCScenario(RefineScenario):
    """REFINE with self-consistency voting on the diagnosis."""

    N_SAMPLES = 3
    TEMPERATURE = 0.7

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,
        diagnostician_config: "ModelConfig | None" = None,
        verifier_config: "ModelConfig | None" = None,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
        self._sc_client, self._sc_model = get_llm("qwen")

    def _diagnose_sc(self, summary: str) -> tuple[str, list[str]]:
        """Sample N diagnoses independently and majority-vote.

        Returns (final_diagnosis, all_samples).
        """
        sys_prompt = self.prompts.get_diagnostician_system_prompt()
        user_prompt = self.prompts.get_diagnostician_instruction(summary)

        samples: list[str] = []
        for _ in range(self.N_SAMPLES):
            try:
                raw = _llm_chat(
                    self._sc_client,
                    self._sc_model,
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.TEMPERATURE,
                    max_tokens=512,
                )
            except Exception:  # pragma: no cover - fail open
                raw = ""
            _, ans = self.extract_action(raw)
            if not ans:
                ans = (raw or "").strip()
            samples.append(ans)

        # Majority vote on normalized diagnosis text.
        normed = [_norm(a) for a in samples]
        top_norm, _ = Counter(normed).most_common(1)[0]
        for a, n in zip(samples, normed):
            if n == top_norm:
                return a, samples
        return samples[0], samples  # unreachable, defensive

    def _run_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer,
        diagnostician,
        verifier,
        doctor,
        patient,
        reporter,
        current_turn: int,
    ) -> tuple[str, bool]:
        """Same pipeline as REFINE, but the diagnosis is self-consistency voted.

        The CAMEL `diagnostician` is not used for the SC sampling; the final
        answer comes from `_diagnose_sc`. The original verifier still runs on
        the voted answer.
        """
        # Summarizer phase
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": summarizer.get_last_duration(),
        })
        summary = self.extract_summary(s_response)

        # Self-consistency diagnosis phase (replaces single diagnostician.step)
        answer, samples = self._diagnose_sc(summary)
        trace.append({
            "role_id": "diagnostician_sc",
            "content": {"voted": answer, "samples": samples},
            "duration": None,
        })

        # Verifier phase (unchanged from REFINE)
        v_instruction = self.prompts.get_verifier_instruction(
            current_turns=current_turn,
            max_turns=self.max_turns,
            summary=summary,
            diagnosis=answer,
        )
        v_response = verifier.step(v_instruction)
        trace.append({
            "role_id": "diagnostician_verifier",
            "content": v_response,
            "duration": verifier.get_last_duration(),
        })

        decision, feedback = self.extract_verifier_decision(v_response)

        if decision == "INCOMPLETE" and current_turn < self.max_turns - 1:
            fb_instruction = self.prompts.get_doctor_feedback_instruction(feedback)
            fb_response = doctor.step(fb_instruction)
            trace.append({
                "role_id": "doctor",
                "content": fb_response,
                "duration": doctor.get_last_duration(),
            })
            action_type, action_content = self.extract_action(fb_response)
            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": reporter.get_last_duration(),
                })
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": patient.get_last_duration(),
                })
            return answer, False

        return answer, True

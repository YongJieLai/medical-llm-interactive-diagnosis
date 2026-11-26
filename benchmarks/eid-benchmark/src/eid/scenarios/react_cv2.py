"""react_cv2: a ReAct-specific CEV-GR plugin.

The generic port (`react_cv` == RefineCVScenario on the ReAct doctor) regressed
6pp on the ReAct base (52.5% -> 46.5%). The mechanism analysis traced it to
three host-mismatches:

1. The ReAct doctor's own [DIAGNOSIS] answer was DISCARDED and replaced by an
   independent diagnostician (so the 52.5% ReAct diagnostician was swapped for a
   weaker summarizer->diagnostician chain on a shallow summary);
2. The [THOUGHT] reasoning never entered the dialogue history (only the bare
   action content did), so the auditor/diagnostician lost the doctor's reasoning;
3. The GATHER feedback used the SC format ("your NEXT action must be [QUERY]/[TEST]"),
   which collides with ReAct's "Reasoning-then-Acting + [DIAGNOSIS] is terminal"
   contract — the ReAct doctor had already committed to a final diagnosis and
   answered the feedback with low-value follow-ups.

`react_cv2` fixes all three while keeping the four-audit CEV-GR loop:

1. candidate diagnosis == the ReAct doctor's OWN [DIAGNOSIS] answer (no separate
   diagnostician);
2. evidence_state == the FULL dialogue history, INCLUDING [THOUGHT] reasoning;
3. verifier == the method-agnostic DiagnosticGate (extracts the key findings
   itself; no [EVIDENCE] structure required);
4. feedback is phrased in ReAct's own contract: REDO -> [THOUGHT]+[DIAGNOSIS]
   (re-diagnose without gathering); BREAK_ANCHOR / GATHER_GAP -> [THOUGHT] then
   ONE [QUERY]/[TEST].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.scenarios.base import CaseInput, ScenarioResult
from eid.scenarios.react import ReactScenario
from eid.scenarios.diagnostic_gate import DiagnosticGate

if TYPE_CHECKING:
    from eid.config import ModelConfig


class ReactCV2Scenario(ReactScenario):
    """ReAct base + ReAct-specific CEV-GR gate (method-agnostic verifier)."""

    MODE = "react_cv2"

    # Cross-model verification, consistent with refine_cv (DeepSeek).
    VERIFIER_PROVIDER = "deepseek"

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,  # noqa: ARG002 - accepted for factory parity, unused
        diagnostician_config: "ModelConfig | None" = None,  # noqa: ARG002
        verifier_config: "ModelConfig | None" = None,  # noqa: ARG002
    ) -> None:
        super().__init__(dataset_name, doctor_config, patient_config, reporter_config, max_turns)
        self._gate = DiagnosticGate(self.prompts, provider=self.VERIFIER_PROVIDER)

    # ------------------------------------------------------------------ #
    # ReAct-format feedback
    # ------------------------------------------------------------------ #
    @staticmethod
    def _feedback_instruction(action: str, feedback: str) -> str:
        """Phrase the gate feedback in ReAct's Reasoning-then-Acting contract."""
        if action == "REDO":
            return (
                "### DIAGNOSTIC FEEDBACK ###\n"
                "A supervisor found that your diagnosis relied on evidence that failed "
                "independent verification. Do NOT gather more data yet — re-examine the "
                "evidence and correct the diagnosis.\n"
                f"Specific Feedback:\n{feedback}\n\n"
                "Follow the Reasoning-then-Acting process:\n"
                "[THOUGHT] re-analyze the evidence WITHOUT the flagged findings\n"
                "[DIAGNOSIS] your corrected diagnosis\n"
                "Doctor:"
            )
        # BREAK_ANCHOR / GATHER_GAP: acquire discriminating evidence.
        return (
            "### DIAGNOSTIC FEEDBACK ###\n"
            "A supervisor reviewed your diagnosis and found that discriminating "
            "information is still missing or unverified. Do NOT give a final diagnosis yet.\n"
            f"Specific Feedback:\n{feedback}\n\n"
            "Follow the Reasoning-then-Acting process:\n"
            "[THOUGHT] analyze what is missing and why it discriminates\n"
            "then execute exactly ONE [QUERY] or [TEST] targeted to acquire it.\n"
            "Doctor:"
        )

    def _route(
        self,
        action_type: str,
        action_content: str,
        patient,
        reporter,
        trace: list[dict],
        dialogue_history: str,
    ) -> tuple[str, str]:
        """Route a doctor action to patient/reporter.

        Returns (last_reply, updated_dialogue_history).
        """
        if action_type == "test":
            m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
            m_response = reporter.step(m_instruction)
            trace.append({
                "role_id": "reporter",
                "content": m_response,
                "duration": reporter.get_last_duration(),
            })
            return m_response, dialogue_history + f"\nMeasurement:\n{m_response}\n"

        p_instruction = self.prompts.get_patient_turn_instruction(action_content)
        p_response = patient.step(p_instruction)
        trace.append({
            "role_id": "patient",
            "content": p_response,
            "duration": patient.get_last_duration(),
        })
        last_reply = self.extract_patient_response(p_response)
        return last_reply, dialogue_history + f"\nPatient:\n{last_reply}\n"

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute the ReAct loop with the ReAct-specific CEV-GR gate."""
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        doctor = create_agent(
            role_id="doctor",
            system_prompt=self.prompts.get_react_doctor_system_prompt().format(
                max_turns=self.max_turns
            ),
            config=self.doctor_config,
            message_window_size=24,
            summarize_threshold=80,
        )
        patient = create_agent(
            role_id="patient",
            system_prompt=self.prompts.get_patient_system_prompt().format(
                patient_facts=patient_facts_str
            ),
            config=self.patient_config,
            message_window_size=1,
            summarize_threshold=95,
        )
        reporter = create_agent(
            role_id="reporter",
            system_prompt=self.prompts.get_reporter_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.reporter_config,
            message_window_size=1,
            summarize_threshold=80,
        )

        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0
        prev_diagnosis: str | None = None
        gather_count = 0

        while current_turn <= self.max_turns:
            instruction = self.prompts.get_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
            )
            doctor_response = doctor.step(instruction)
            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": doctor.get_last_duration(),
            })
            action_type, action_content = self.extract_action(doctor_response)
            # Keep the FULL doctor turn (incl. [THOUGHT]) in the evidence state.
            dialogue_history += f"\nDoctor:\n{doctor_response}\n"

            if action_type == "diagnosis":
                decision = self._gate.decide(
                    evidence_state=dialogue_history,
                    diagnosis=action_content,
                    prev_diagnosis=prev_diagnosis,
                    current_turn=current_turn,
                    max_turns=self.max_turns,
                )
                trace.append({
                    "role_id": "verifier",
                    "content": decision.verification,
                    "duration": None,
                })

                if decision.action == "PASS":
                    answer = action_content
                    break

                prev_diagnosis = action_content

                if decision.action == "REDO":
                    # Re-diagnose WITHOUT gathering (evidence is broken).
                    fb = self._feedback_instruction("REDO", decision.feedback)
                    redo_response = doctor.step(fb)
                    trace.append({
                        "role_id": "doctor",
                        "content": redo_response,
                        "duration": doctor.get_last_duration(),
                    })
                    dialogue_history += f"\nDoctor:\n{redo_response}\n"
                    rt, rc = self.extract_action(redo_response)
                    current_turn += 1
                    if rt == "diagnosis":
                        # Accept the corrected diagnosis directly (no second gate,
                        # to avoid an unbounded redo loop).
                        answer = rc
                        break
                    last_reply, dialogue_history = self._route(
                        rt, rc, patient, reporter, trace, dialogue_history
                    )
                    continue

                # BREAK_ANCHOR / GATHER_GAP: acquire the missing evidence.
                if gather_count < 2 and current_turn < self.max_turns - 1:
                    gather_count += 1
                    fb = self._feedback_instruction(decision.action, decision.feedback)
                    fb_response = doctor.step(fb)
                    trace.append({
                        "role_id": "doctor",
                        "content": fb_response,
                        "duration": doctor.get_last_duration(),
                    })
                    dialogue_history += f"\nDoctor:\n{fb_response}\n"
                    fa, fc = self.extract_action(fb_response)
                    current_turn += 1
                    last_reply, dialogue_history = self._route(
                        fa, fc, patient, reporter, trace, dialogue_history
                    )
                    continue

                # Gather cap reached / out of turns: accept the current diagnosis.
                answer = action_content
                break

            # Non-diagnosis: ordinary gather turn.
            current_turn += 1
            last_reply, dialogue_history = self._route(
                action_type, action_content, patient, reporter, trace, dialogue_history
            )

        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("reporter", reporter),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": self.MODE,
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
            },
        )

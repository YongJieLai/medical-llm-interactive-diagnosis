"""REFINE-EG scenario (Evidence-Grounded REFINE).

A zero-training plugin on REFINE that keeps the original
summarizer -> diagnostician -> verifier structure (free-reasoning
diagnostician is preserved), but adds evidence-discipline to the
diagnostician and a consistency check to the verifier.

This is the corrected design after the REFINE-HPR negative result:
HPR inserted an explicit differential-diagnosis LIST between summarizer and
diagnostician, which shifted the decision onto an unstable list generator and
caused regressions. REFINE-EG instead bakes the evidence rules directly into
the diagnostician prompt, so the model keeps its full reasoning ability while
being prevented from hallucinating tests / misreading signs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper


class RefineEGScenario(BaseScenario):
    """REFINE with evidence-grounded diagnostician and consistency verifier."""

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
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.summarizer_config = summarizer_config or doctor_config
        self.diagnostician_config = diagnostician_config or doctor_config
        self.verifier_config = verifier_config or doctor_config
        self.prompts = PromptManager(dataset_name)

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        doctor = create_agent(
            role_id="doctor",
            system_prompt=self.prompts.get_sc_doctor_system_prompt().format(
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

        summarizer = create_agent(
            role_id="summarizer",
            system_prompt=self.prompts.get_summarizer_system_prompt(),
            config=self.summarizer_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        diagnostician = create_agent(
            role_id="diagnostician",
            system_prompt=self.prompts.get_diagnostician_eg_system_prompt(),
            config=self.diagnostician_config,
            message_window_size=3,
            summarize_threshold=90,
        )

        verifier = create_agent(
            role_id="diagnostician_verifier",
            system_prompt=self.prompts.get_verifier_hpr_system_prompt(),
            config=self.verifier_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0

        while current_turn <= self.max_turns:
            instruction = self.prompts.get_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
            )

            doctor_response = doctor.step(instruction)
            duration = doctor.get_last_duration()
            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": duration,
            })

            action_type, action_content = self.extract_action(doctor_response)
            dialogue_history += f"\nDoctor: {action_content}\n"

            if action_type == "finish" or current_turn >= self.max_turns:
                answer, verified = self._run_verification_pipeline(
                    trace=trace,
                    dialogue_history=dialogue_history,
                    summarizer=summarizer,
                    diagnostician=diagnostician,
                    verifier=verifier,
                    doctor=doctor,
                    patient=patient,
                    reporter=reporter,
                    current_turn=current_turn,
                )

                if verified or current_turn >= self.max_turns - 1:
                    break

                current_turn += 1
                continue

            current_turn += 1

            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()
                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"

            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()
                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })
                last_reply = self.extract_patient_response(p_response)
                dialogue_history += f"\nPatient: {last_reply}\n"

        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("reporter", reporter),
            ("summarizer", summarizer),
            ("diagnostician", diagnostician),
            ("verifier", verifier),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "refine_eg",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
            },
        )

    def _run_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        verifier: "AgentWrapper",
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        current_turn: int,
    ) -> tuple[str, bool]:
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        s_duration = summarizer.get_last_duration()
        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": s_duration,
        })
        summary = self.extract_summary(s_response)

        d_instruction = self.prompts.get_diagnostician_instruction(summary)
        d_response = diagnostician.step(d_instruction)
        d_duration = diagnostician.get_last_duration()
        trace.append({
            "role_id": "diagnostician",
            "content": d_response,
            "duration": d_duration,
        })

        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response

        v_instruction = self.prompts.get_verifier_instruction(
            current_turns=current_turn,
            max_turns=self.max_turns,
            summary=summary,
            diagnosis=answer,
        )
        v_response = verifier.step(v_instruction)
        v_duration = verifier.get_last_duration()
        trace.append({
            "role_id": "diagnostician_verifier",
            "content": v_response,
            "duration": v_duration,
        })

        decision, feedback = self.extract_verifier_decision(v_response)

        if decision == "INCOMPLETE" and current_turn < self.max_turns - 1:
            fb_instruction = self.prompts.get_doctor_feedback_instruction(feedback)
            fb_response = doctor.step(fb_instruction)
            fb_duration = doctor.get_last_duration()
            trace.append({
                "role_id": "doctor",
                "content": fb_response,
                "duration": fb_duration,
            })

            action_type, action_content = self.extract_action(fb_response)

            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()
                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()
                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })

            return answer, False

        return answer, True

    def _collect_role_records(
        self, agents: list[tuple[str, "AgentWrapper"]]
    ) -> list[dict]:
        records = []
        for role_id, agent in agents:
            history = agent.get_history()
            token_usage = agent.get_usage_stats()
            records.append({
                role_id: history,
                "token_usage": token_usage,
            })
        return records

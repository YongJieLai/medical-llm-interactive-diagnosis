"""MEDDxAgent scenario (mechanism-level port).

Ports the core orchestration mechanism of MEDDxAgent (NEC, ACL 2025)
"DDxDriver" onto the EID / AgentClinic environment.

Faithful parts kept from the original:
1. Modular agent pool: each turn the doctor decides which module to invoke
   (HISTORY_TAKING vs EXAMINATION vs DIAGNOSIS), rather than free-form chatting.
2. Incremental Patient Profile: after each exchange, a scribe agent merges the
   new information into a structured working profile (the original calls this
   `dialogue_to_patient_profile`), which is what the doctor reads each turn
   instead of the raw transcript.
3. Forced final diagnosis: on the last turn the doctor MUST issue a diagnosis.

The RAG (retrieval) module of the original is dropped because AgentClinic has
no external knowledge base wired in; EXAMINATION (test ordering) plays the
role of acquiring objective evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper


DOCTOR_SYSTEM_PROMPT = """You are a DDxDriver: a modular differential-diagnosis agent that iteratively refines a diagnosis by routing each turn through a dedicated sub-module.

You read a working Patient Profile that accumulates all clinical information gathered so far. Do not re-derive facts from scratch; use the profile as your working state.

Available modules (invoke exactly ONE per turn):
1. HISTORY_TAKING  -> ask the patient ONE atomic question to fill a specific information gap.
2. EXAMINATION     -> request ONE clinical examination or laboratory test (answered by the Measurement module).
3. DIAGNOSIS       -> synthesize the Patient Profile into the single most likely diagnosis.

Operational constraints:
1. Turn Limit: you strictly cannot exceed {max_turns} total turns.
2. No Repetition: never ask a question or request a test already covered.
3. Atomic Inquiries: each question addresses a single, specific topic.

Every turn you execute exactly ONE action in the corresponding format:
1. [QUERY] followed by your atomic question to the patient.
2. [TEST] followed by one specific examination or diagnostic test request.
3. [DIAGNOSIS] followed by the single most likely diagnosis.

Once you have gathered sufficient evidence, ensure your diagnosis is final."""


PROFILE_UPDATER_SYSTEM_PROMPT = """You are a clinical scribe maintaining a structured Patient Profile used for differential diagnosis.

Core principles:
1. Strict adherence: you must NOT invent, infer, or hallucinate any information not explicitly present in the input.
2. Precision: retain all precise measurements, timings, and medical terms exactly as stated.
3. Accumulation: preserve all previously collected information; the profile is monotonic (you only add/refine, never drop).

Output format:
[PROFILE] followed by the updated structured patient profile."""


def _doctor_turn_instruction(
    current_turns: int,
    max_turns: int,
    patient_profile: str,
    last_reply: str,
    gate_feedback: str = "",
) -> str:
    force = ""
    if current_turns >= max_turns:
        force = "\nThis is the FINAL turn. You MUST issue a [DIAGNOSIS] now with your single most likely diagnosis."
    fb = ""
    if gate_feedback:
        fb = (
            "\n\n### SUPERVISOR FEEDBACK (address this with your next action) ###\n"
            f"{gate_feedback}\n"
            "Respond with ONE action: a targeted [QUERY]/[TEST] to acquire the missing or "
            "verification evidence, or a corrected [DIAGNOSIS]."
        )
    return (
        f"Turns used: {current_turns} / {max_turns}.\n"
        f"Current Patient Profile:\n{patient_profile}\n\n"
        f"Last reply:\n{last_reply}\n"
        f"{fb}"
        f"{force}\n\n"
        "Doctor:"
    )


def _profile_update_instruction(old_profile: str, new_exchange: str) -> str:
    return (
        "Existing Patient Profile:\n"
        f"{old_profile}\n\n"
        "New information from the latest exchange:\n"
        f"{new_exchange}\n\n"
        "Merge the new information into the profile and output the updated [PROFILE]."
    )


class MEDDxAgentScenario(BaseScenario):
    """MEDDxAgent-style modular DDxDriver scenario."""

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
    ) -> None:
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.prompts = PromptManager(dataset_name)
        # Method-agnostic verification gate. None = baseline (no gate).
        # Subclasses set this to a DiagnosticGate to enable the plugin.
        self.gate = None

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        doctor = create_agent(
            role_id="doctor",
            system_prompt=DOCTOR_SYSTEM_PROMPT.format(max_turns=self.max_turns),
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

        profile_updater = create_agent(
            role_id="profile_updater",
            system_prompt=PROFILE_UPDATER_SYSTEM_PROMPT,
            config=self.doctor_config,
            message_window_size=2,
            summarize_threshold=90,
        )

        trace: list[dict] = []
        patient_profile = "(empty — no information gathered yet)"
        last_reply = ""
        answer = ""
        current_turn = 0
        prev_diagnosis: str | None = None
        gate_feedback = ""

        while current_turn <= self.max_turns:
            instruction = _doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                patient_profile=patient_profile,
                last_reply=last_reply,
                gate_feedback=gate_feedback,
            )

            doctor_response = doctor.step(instruction)
            duration = doctor.get_last_duration()

            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": duration,
            })

            action_type, action_content = self.extract_action(doctor_response)

            if action_type == "diagnosis":
                if self.gate is None:
                    answer = action_content
                    break

                candidate = action_content
                decision = self.gate.decide(
                    evidence_state=patient_profile,
                    diagnosis=candidate,
                    prev_diagnosis=prev_diagnosis,
                    current_turn=current_turn,
                    max_turns=self.max_turns,
                )
                trace.append({
                    "role_id": "gate",
                    "content": decision.verification,
                    "duration": None,
                })

                if decision.action == "PASS":
                    answer = candidate
                    break

                # Rejected: record for anchoring check, feed back, keep going.
                prev_diagnosis = candidate
                gate_feedback = decision.feedback
                current_turn += 1
                continue

            current_turn += 1

            # Route to simulator
            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()

                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
                new_exchange = f"Doctor requested: {action_content}\nMeasurement: {m_response}"
                last_reply = m_response

            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })
                patient_reply = self.extract_patient_response(p_response)
                new_exchange = f"Doctor asked: {action_content}\nPatient: {patient_reply}"
                last_reply = patient_reply

            # Incremental Patient Profile update (MEDDxAgent core mechanism)
            pu_instruction = _profile_update_instruction(patient_profile, new_exchange)
            pu_response = profile_updater.step(pu_instruction)
            pu_duration = profile_updater.get_last_duration()

            trace.append({
                "role_id": "profile_updater",
                "content": pu_response,
                "duration": pu_duration,
            })

            match = __import__("re").search(r"\[PROFILE\]\s*([\s\S]*)", pu_response)
            if match:
                patient_profile = match.group(1).strip()

        # Fallback: if the gate kept rejecting until the turn budget ran out,
        # fall back to the last candidate diagnosis (it is still the host's
        # best answer, just not gated-PASS).
        if not answer and prev_diagnosis:
            answer = prev_diagnosis

        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("reporter", reporter),
            ("profile_updater", profile_updater),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "meddxagent_gate" if self.gate is not None else "meddxagent",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
            },
        )

    def _collect_role_records(self, agents: list[tuple[str, "AgentWrapper"]]) -> list[dict]:
        records = []
        for role_id, agent in agents:
            history = agent.get_history()
            token_usage = agent.get_usage_stats()
            records.append({
                role_id: history,
                "token_usage": token_usage,
            })
        return records

"""ClinicalAgents scenario (mechanism-level port).

Ports the core orchestration mechanism of ClinicalAgents (KDD 2026)
onto the EID / AgentClinic environment.

Faithful parts kept from the original:
1. Explicit Working Memory: a structured, mutable state (EVIDENCE / HYPOTHESES /
   MISSING evidence / action trace) that is updated every turn, instead of the
   raw transcript being the only state.
2. Hypothesis-driven evidence acquisition: each turn the orchestrator first
   re-forms its differential diagnosis, then detects the critical missing
   evidence, and only then chooses an action.
3. Backtracking: when evidence fails to support the leading hypothesis, the
   orchestrator must drop it and gather evidence for alternatives (the original
   triggers "backtrack to the correct workflow stage" on missing evidence).

Deliberate simplifications (documented, not hidden):
- The original's MCTS tree search (Top-K actions x N rollouts) is replaced by a
  single-step LLM decision, because AgentClinic's action space is only 3
  (QUERY/TEST/DIAGNOSIS) and full rollouts are cost-prohibitive via API. The
  "hypothesis -> missing-evidence -> backtrack" loop, which the paper's ablation
  shows is the main source of gain, is kept intact.
- The original's Experience Memory (guideline + historical-case retrieval) is
  dropped because AgentClinic has no external knowledge base; the orchestrator
  relies on its internal knowledge.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper


ORCHESTRATOR_SYSTEM_PROMPT = """You are a clinical orchestration agent performing hypothesis-driven diagnosis with backtracking.

You maintain a Working Memory: a structured, mutable state with four components:
- EVIDENCE: all clinical findings gathered so far.
- HYPOTHESES: your current differential diagnosis, each with a confidence level.
- MISSING: the critical missing evidence that would discriminate between the leading hypotheses.
- ACTIONS: the action history (what has already been asked/ordered).

Each turn, follow this cognitive cycle:
1. Output [MEMORY] with your UPDATED Working Memory (EVIDENCE / HYPOTHESES / MISSING / ACTIONS).
2. Then execute exactly ONE action:
   - [QUERY] followed by one atomic question targeting a specific MISSING item.
   - [TEST] followed by one specific examination or diagnostic test request.
   - [DIAGNOSIS] followed by the single most likely diagnosis.

Backtracking rule: if collected evidence contradicts or fails to support a leading
hypothesis, explicitly remove it from HYPOTHESES and pivot to gathering evidence
for the remaining alternatives.

Operational constraints:
1. Turn Limit: strictly cannot exceed {max_turns} total turns.
2. No Repetition: never re-ask or re-order something already in ACTIONS.
3. Atomic Inquiries: one specific topic per question.
4. Evidence-grounded: a hypothesis without supporting evidence must not survive."""


def _orchestrator_turn_instruction(current_turns: int, max_turns: int, working_memory: str, last_reply: str) -> str:
    force = ""
    if current_turns >= max_turns:
        force = "\nThis is the FINAL turn. You MUST issue a [DIAGNOSIS] now with your single most likely diagnosis."
    return (
        f"Turns used: {current_turns} / {max_turns}.\n"
        f"Working Memory:\n{working_memory}\n\n"
        f"Last reply:\n{last_reply}\n"
        f"{force}\n\n"
        "Orchestrator:"
    )


def _extract_memory(text: str) -> str:
    m = re.search(r"\[MEMORY\]\s*([\s\S]*?)(?=\[QUERY\]|\[TEST\]|\[DIAGNOSIS\]|\[FINISH\]|$)", text)
    if m:
        return m.group(1).strip()
    return ""


class ClinicalAgentsScenario(BaseScenario):
    """ClinicalAgents-style MCTS-orchestration scenario (single-step LLM decision)."""

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

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        orchestrator = create_agent(
            role_id="orchestrator",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT.format(max_turns=self.max_turns),
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
        working_memory = "EVIDENCE: (none)\nHYPOTHESES: (none)\nMISSING: (unknown)\nACTIONS: (none)"
        last_reply = ""
        answer = ""
        current_turn = 0

        while current_turn <= self.max_turns:
            instruction = _orchestrator_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                working_memory=working_memory,
                last_reply=last_reply,
            )

            orch_response = orchestrator.step(instruction)
            duration = orchestrator.get_last_duration()

            trace.append({
                "role_id": "orchestrator",
                "content": orch_response,
                "duration": duration,
            })

            # Update explicit Working Memory from [MEMORY] tag
            mem = _extract_memory(orch_response)
            if mem:
                working_memory = mem

            action_type, action_content = self.extract_action(orch_response)

            if action_type == "diagnosis":
                answer = action_content
                break

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

        role_records = self._collect_role_records([
            ("orchestrator", orchestrator),
            ("patient", patient),
            ("reporter", reporter),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "clinicalagents",
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

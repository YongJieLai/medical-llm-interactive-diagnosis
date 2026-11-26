"""CEV-GR ported onto alternative host-method bases (React / Roleplay).

The CEV-GR plugin is the four-check verification loop in `RefineCVScenario`
(claim check / gap check / refutation / decisiveness gate + GATHER re-loop).
That loop is host-agnostic: it only needs a doctor/patient/reporter/summarizer/
diagnostician and a `dialogue_history`. To port it onto a different host method
we only override two hooks:

  - `_doctor_system_prompt()`  -> the host method's doctor prompt
  - `_is_finish_action()`      -> the host method's "I'm done" action

The host method's OWN diagnostician is NOT reused: CEV-GR requires the
auditable CV diagnostician (which emits [EVIDENCE]) so the claim check has
statements to audit. This is part of the plugin, not the host base.

Note on SC: `RefineCVScenario` already uses the SC doctor (`get_sc_doctor_system_prompt`),
so "SC + CEV-GR" == `refine_cv` and is therefore NOT duplicated here. The genuinely
new port is React (ReAct doctor + [DIAGNOSIS] terminal action).
"""

from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario


class ReactCVScenario(RefineCVScenario):
    """React (ReAct) base + CEV-GR four-check verification loop.

    The ReAct doctor emits [THOUGHT] then [QUERY]/[TEST]/[DIAGNOSIS]. Its
    terminal action is [DIAGNOSIS] (there is no [FINISH] command), so we
    trigger the verification pipeline on `action_type == "diagnosis"`.
    """

    MODE = "react_cv"

    def _doctor_system_prompt(self) -> str:
        return self.prompts.get_react_doctor_system_prompt()

    def _is_finish_action(self, action_type: str) -> bool:
        return action_type == "diagnosis"


class RoleplayCVScenario(RefineCVScenario):
    """Roleplay (base doctor) + CEV-GR four-check verification loop.

    The base doctor emits [QUERY]/[TEST]/[DIAGNOSIS] with no [FINISH], so the
    terminal action is [DIAGNOSIS].
    """

    MODE = "roleplay_cv"

    def _doctor_system_prompt(self) -> str:
        return self.prompts.get_base_doctor_system_prompt()

    def _is_finish_action(self, action_type: str) -> bool:
        return action_type == "diagnosis"

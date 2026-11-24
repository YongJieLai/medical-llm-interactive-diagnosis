"""MEDDxAgent + DiagnosticGate: the method-agnostic plugin applied to MEDDxAgent.

This is the first "method-agnostic plugin" validation host. It subclasses the
baseline MEDDxAgentScenario and only enables the DiagnosticGate (same-model
verifier). Everything else (modular agent pool, incremental patient profile,
forced final diagnosis) is byte-for-byte the baseline.
"""

from __future__ import annotations

from eid.scenarios.diagnostic_gate import DiagnosticGate
from eid.scenarios.meddxagent import MEDDxAgentScenario


class MEDDxAgentGateScenario(MEDDxAgentScenario):
    """MEDDxAgent with the verification gate inserted at the diagnosis step."""

    # Same-model verification (cross-model semantic misalignment collapsed to 40%).
    GATE_PROVIDER = "qwen"

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config, max_turns=16):
        super().__init__(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
        )
        self.gate = DiagnosticGate(self.prompts, provider=self.GATE_PROVIDER)

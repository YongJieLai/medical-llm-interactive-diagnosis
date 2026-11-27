"""Evaluation scenarios module.

Each scenario defines a specific evaluation mode with its own
interaction pattern between agents.
"""

from eid.scenarios.base import BaseScenario, ScenarioResult
from eid.scenarios.cot import CoTScenario
from eid.scenarios.roleplay import RoleplayScenario
from eid.scenarios.react import ReactScenario
from eid.scenarios.sc import SCScenario
from eid.scenarios.refine import RefineScenario
from eid.scenarios.refine_hpr import RefineHPRScenario
from eid.scenarios.refine_eg import RefineEGScenario
from eid.scenarios.refine_cv import RefineCVScenario
from eid.scenarios.refine_cv_p01 import (
    RefineCVP0Scenario,
    RefineCVP1Scenario,
    RefineCVP0P1Scenario,
)
from eid.scenarios.refine_cv_staged import (
    RefineCVStagedScenario,
    RefineCVStagedK4Scenario,
    RefineCVStagedK8Scenario,
)
from eid.scenarios.refine_cv_cover import RefineCVCoverScenario
from eid.scenarios.refine_cover import RefineCoverScenario
from eid.scenarios.refine_cv_freebudget import RefineCVFreebudgetScenario
from eid.scenarios.refine_cv_cover_plan import RefineCVCoverPlanScenario
from eid.scenarios.refine_cv_debias import (
    RefineCVDebiasScenario,
    RefineCVCoverDebiasScenario,
)
from eid.scenarios.refine_cv_discrim import RefineCVDiscrimScenario
from eid.scenarios.refine_cv_ports import ReactCVScenario, RoleplayCVScenario
from eid.scenarios.refine_cv_cover_ports import ReactCoverScenario, RoleplayCoverScenario
from eid.scenarios.roleplay_cover_abl import RoleplayCoverAblScenario
from eid.scenarios.sc_cover import SCCoverScenario
from eid.scenarios.react_cv2 import ReactCV2Scenario
from eid.scenarios.refine_cv_qwen import RefineCVQwenScenario
from eid.scenarios.refine_ht import RefineHTScenario
from eid.scenarios.refine_cv_primekg import RefineCVQwenGapRefutePrimeKGScenario
from eid.scenarios.refine_cv_runtimeddx import RefineCVQwenGapRefuteLLMDDXScenario
from eid.scenarios.refine_cv_emptyddx import RefineCVQwenGapRefuteEmptyDDXScenario
from eid.scenarios.refine_cv_doctorddx import RefineCVQwenGapRefuteDoctorDDXScenario
from eid.scenarios.refine_cv_ddxdiag import (
    RefineCVQwenGapRefuteDDXDiagScenario,
    RefineCVQwenGapRefuteNeutralSummScenario,
    RefineCVQwenGapRefuteJointScenario,
)
from eid.scenarios.refine_cv_llmddx_variants import (
    RefineCVQwenGapRefuteLLMDDXNeutralSummScenario,
    RefineCVQwenGapRefuteLLMDDXQwenGenScenario,
)
from eid.scenarios.refine_cv_ablations import (
    RefineCVQwenCheckOnlyScenario,
    RefineCVQwenGapOnlyScenario,
    RefineCVQwenRefuteOnlyScenario,
    RefineCVQwenGapRefuteScenario,
    RefineCVQwenGapRefuteRAGScenario,
    RefineCVQwenRAGScenario,
    RefineCVQwenRefuteOnlyRAGScenario,
    RefineCVQwenMaxScenario,
    RefineCVGLMScenario,
)
from eid.scenarios.refine_sc import RefineSCScenario
from eid.scenarios.meddxagent import MEDDxAgentScenario
from eid.scenarios.meddxagent_gate import MEDDxAgentGateScenario
from eid.scenarios.clinicalagents import ClinicalAgentsScenario

__all__ = [
    "BaseScenario",
    "ScenarioResult",
    "CoTScenario",
    "RoleplayScenario",
    "ReactScenario",
    "SCScenario",
    "RefineScenario",
    "RefineHPRScenario",
    "RefineEGScenario",
    "RefineCVScenario",
    "RefineCVP0Scenario",
    "RefineCVP1Scenario",
    "RefineCVP0P1Scenario",
    "RefineCVStagedScenario",
    "RefineCVStagedK4Scenario",
    "RefineCVStagedK8Scenario",
    "RefineCVCoverScenario",
    "RefineCVFreebudgetScenario",
    "RefineCVCoverPlanScenario",
    "RefineCVDebiasScenario",
    "RefineCVCoverDebiasScenario",
    "RefineCVDiscrimScenario",
    "RefineCVQwenScenario",
    "ReactCVScenario",
    "RoleplayCVScenario",
    "ReactCV2Scenario",
    "RefineHTScenario",
    "RefineCVQwenCheckOnlyScenario",
    "RefineCVQwenGapOnlyScenario",
    "RefineCVQwenRefuteOnlyScenario",
    "RefineCVQwenGapRefuteScenario",
    "RefineCVQwenGapRefuteRAGScenario",
    "RefineCVQwenGapRefutePrimeKGScenario",
    "RefineCVQwenGapRefuteLLMDDXScenario",
    "RefineCVQwenGapRefuteEmptyDDXScenario",
    "RefineCVQwenGapRefuteDoctorDDXScenario",
    "RefineCVQwenGapRefuteDDXDiagScenario",
    "RefineCVQwenGapRefuteNeutralSummScenario",
    "RefineCVQwenGapRefuteJointScenario",
    "RefineCVQwenGapRefuteLLMDDXNeutralSummScenario",
    "RefineCVQwenGapRefuteLLMDDXQwenGenScenario",
    "RefineCVQwenRAGScenario",
    "RefineCVQwenRefuteOnlyRAGScenario",
    "RefineCVQwenMaxScenario",
    "RefineCVGLMScenario",
    "RefineSCScenario",
    "MEDDxAgentScenario",
    "MEDDxAgentGateScenario",
    "ClinicalAgentsScenario",
    "get_scenario",
]


def get_scenario(
    mode: str,
    dataset_name: str,
    doctor_config: "ModelConfig",  # noqa: F821
    patient_config: "ModelConfig | None" = None,  # noqa: F821
    reporter_config: "ModelConfig | None" = None,  # noqa: F821
    max_turns: int = 16,
    summarizer_config: "ModelConfig | None" = None,  # noqa: F821
    diagnostician_config: "ModelConfig | None" = None,  # noqa: F821
    verifier_config: "ModelConfig | None" = None,  # noqa: F821
) -> BaseScenario:
    """Factory function to create a scenario by mode name.

    Args:
        mode: Scenario mode (cot, roleplay, react, sc, refine)
        dataset_name: Name of the dataset
        doctor_config: Model config for doctor role
        patient_config: Model config for patient simulator
        reporter_config: Model config for reporter simulator
        max_turns: Maximum interaction turns
        summarizer_config: Model config for summarizer role (SC/REFINE)
        diagnostician_config: Model config for diagnostician role (SC/REFINE)
        verifier_config: Model config for verifier role (REFINE)

    Returns:
        Configured scenario instance

    Raises:
        ValueError: If mode is not recognized
    """
    scenarios = {
        "cot": CoTScenario,
        "roleplay": RoleplayScenario,
        "react": ReactScenario,
        "sc": SCScenario,
        "refine": RefineScenario,
        "refine_hpr": RefineHPRScenario,
        "refine_eg": RefineEGScenario,
        "refine_cv": RefineCVScenario,
        "refine_cv_p0": RefineCVP0Scenario,
        "refine_cv_p1": RefineCVP1Scenario,
        "refine_cv_p0p1": RefineCVP0P1Scenario,
        "refine_cv_staged": RefineCVStagedScenario,
        "refine_cv_staged_k4": RefineCVStagedK4Scenario,
        "refine_cv_staged_k8": RefineCVStagedK8Scenario,
        "refine_cv_cover": RefineCVCoverScenario,
        "refine_cover": RefineCoverScenario,
        "refine_cv_freebudget": RefineCVFreebudgetScenario,
        "refine_cv_cover_plan": RefineCVCoverPlanScenario,
        "refine_cv_debias": RefineCVDebiasScenario,
        "refine_cv_cover_debias": RefineCVCoverDebiasScenario,
        "refine_cv_discrim": RefineCVDiscrimScenario,
        "refine_cv_qwen": RefineCVQwenScenario,
        "react_cv": ReactCVScenario,
        "react_cv2": ReactCV2Scenario,
        "roleplay_cv": RoleplayCVScenario,
        "react_cover": ReactCoverScenario,
        "roleplay_cover": RoleplayCoverScenario,
        "roleplay_cover_abl": RoleplayCoverAblScenario,
        "sc_cover": SCCoverScenario,
        "refine_ht": RefineHTScenario,
        "refine_cv_qwen_check": RefineCVQwenCheckOnlyScenario,
        "refine_cv_qwen_gap": RefineCVQwenGapOnlyScenario,
        "refine_cv_qwen_refute": RefineCVQwenRefuteOnlyScenario,
        "refine_cv_qwen_gap_refute": RefineCVQwenGapRefuteScenario,
        "refine_cv_qwen_gap_refute_rag": RefineCVQwenGapRefuteRAGScenario,
        "refine_cv_qwen_rag": RefineCVQwenRAGScenario,
        "refine_cv_qwen_refute_rag": RefineCVQwenRefuteOnlyRAGScenario,
        "refine_cv_qwenmax": RefineCVQwenMaxScenario,
        "refine_cv_glm": RefineCVGLMScenario,
        "refine_sc": RefineSCScenario,
        "meddxagent": MEDDxAgentScenario,
        "meddxagent_gate": MEDDxAgentGateScenario,
        "clinicalagents": ClinicalAgentsScenario,
        "refine_cv_qwen_gap_refute_primekg": RefineCVQwenGapRefutePrimeKGScenario,
        "refine_cv_qwen_gap_refute_llmddx": RefineCVQwenGapRefuteLLMDDXScenario,
        "refine_cv_qwen_gap_refute_emptyddx": RefineCVQwenGapRefuteEmptyDDXScenario,
        "refine_cv_qwen_gap_refute_doctorddx": RefineCVQwenGapRefuteDoctorDDXScenario,
        "refine_cv_qwen_gap_refute_ddxdiag": RefineCVQwenGapRefuteDDXDiagScenario,
        "refine_cv_qwen_gap_refute_neutralsumm": RefineCVQwenGapRefuteNeutralSummScenario,
        "refine_cv_qwen_gap_refute_joint": RefineCVQwenGapRefuteJointScenario,
        "refine_cv_qwen_gap_refute_llmddx_neutralsumm": RefineCVQwenGapRefuteLLMDDXNeutralSummScenario,
        "refine_cv_qwen_gap_refute_llmddx_qwengen": RefineCVQwenGapRefuteLLMDDXQwenGenScenario,
    }

    if mode not in scenarios:
        raise ValueError(f"Unknown scenario mode: {mode}. Available: {list(scenarios.keys())}")

    scenario_class = scenarios[mode]

    if mode == "cot":
        return scenario_class(dataset_name=dataset_name, doctor_config=doctor_config)

    # For roleplay modes, ensure simulator configs are provided
    if patient_config is None:
        patient_config = doctor_config
    if reporter_config is None:
        reporter_config = doctor_config

    # SC and REFINE modes need additional configs
    if mode in ("sc", "sc_cover"):
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
        )
    elif mode == "refine":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode == "refine_hpr":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode == "refine_eg":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode in ("refine_cv", "refine_cv_p0", "refine_cv_p1", "refine_cv_p0p1", "refine_cv_staged", "refine_cv_staged_k4", "refine_cv_staged_k8", "react_cv", "react_cv2", "roleplay_cv", "react_cover", "roleplay_cover"):
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode in ("refine_cv_qwen", "refine_ht"):
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode in (
        "refine_cv_qwen_check",
        "refine_cv_qwen_gap",
        "refine_cv_qwen_refute",
        "refine_cv_qwen_gap_refute",
        "refine_cv_qwen_gap_refute_rag",
        "refine_cv_qwen_gap_refute_primekg",
        "refine_cv_qwen_gap_refute_llmddx",
        "refine_cv_qwen_gap_refute_emptyddx",
        "refine_cv_qwen_gap_refute_doctorddx",
        "refine_cv_qwen_gap_refute_ddxdiag",
        "refine_cv_qwen_gap_refute_neutralsumm",
        "refine_cv_qwen_gap_refute_joint",
        "refine_cv_qwen_gap_refute_llmddx_neutralsumm",
        "refine_cv_qwen_gap_refute_llmddx_qwengen",
        "refine_cv_qwenmax",
        "refine_cv_glm",
    ):
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )
    elif mode == "refine_sc":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            reporter_config=reporter_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )

    return scenario_class(
        dataset_name=dataset_name,
        doctor_config=doctor_config,
        patient_config=patient_config,
        reporter_config=reporter_config,
        max_turns=max_turns,
    )

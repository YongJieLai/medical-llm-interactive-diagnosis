"""REFINE-CV-QWEN ablation: same-model verification.

Identical to REFINE-CV except the verifier runs on the SAME provider as the
diagnostician (Qwen) instead of DeepSeek. This isolates the contribution of
cross-model decoupling (mechanism ①): if same-model verification performs as
well as cross-model, then the "must decouple" claim is falsified.
"""

from eid.scenarios.refine_cv import RefineCVScenario


class RefineCVQwenScenario(RefineCVScenario):
    """REFINE-CV with a same-model (Qwen) verifier."""

    VERIFIER_PROVIDER = "qwen"

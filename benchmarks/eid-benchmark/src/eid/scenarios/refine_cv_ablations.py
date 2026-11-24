"""Component ablations and robustness variants of REFINE-CV.

P0 (component ablation) — isolate which of the three mechanisms drives the
same-model-verification gain (refine_cv_qwen = 73.3% vs refine = 60.0%):

- RefineCVQwenCheckOnlyScenario  : keep CLAIM CHECK only (drop gap + refutation)
- RefineCVQwenGapOnlyScenario    : keep GAP CHECK only   (drop claim + refutation)
- RefineCVQwenRefuteOnlyScenario : keep REFUTATION only  (drop claim + gap)

P1 (robustness) — rule out the "DeepSeek is just a weaker model" alternative
explanation for why cross-model verification (refine_cv = 40%) failed:

- RefineCVQwenMaxScenario : qwen-plus generation + qwen-MAX verification
                            (cross-model with a STRONGER verifier). If this still
                            underperforms same-model, "cross-model misalignment"
                            holds; if it recovers, DeepSeek was simply too weak.

The "all-DeepSeek" robustness variant (DeepSeek generation + DeepSeek
verification) is NOT a new scenario class: it is RefineCVScenario run with a
DeepSeek doctor_config, so generation and verification share the same model.
"""

from eid.scenarios.refine_cv import RefineCVScenario
from eid.scenarios.refine_cv_qwen import RefineCVQwenScenario


class RefineCVQwenCheckOnlyScenario(RefineCVQwenScenario):
    """Same-model verification with CLAIM CHECK only."""

    ENABLE_GAP_CHECK = False
    ENABLE_REFUTATION = False


class RefineCVQwenGapOnlyScenario(RefineCVQwenScenario):
    """Same-model verification with GAP CHECK only."""

    ENABLE_CLAIM_CHECK = False
    ENABLE_REFUTATION = False


class RefineCVQwenRefuteOnlyScenario(RefineCVQwenScenario):
    """Same-model verification with REFUTATION trigger only."""

    ENABLE_CLAIM_CHECK = False
    ENABLE_GAP_CHECK = False


class RefineCVQwenMaxScenario(RefineCVScenario):
    """Cross-model verification with a STRONGER verifier (qwen-max)."""

    VERIFIER_PROVIDER = "qwen"
    VERIFIER_MODEL = "qwen-max"


class RefineCVGLMScenario(RefineCVScenario):
    """Verifier runs on GLM (Zhipu), a THIRD model family for F3 generalization.

    Combined with --doctor glm / qwen / deepseek, this yields the cross-family
    pairs (qwen→glm, deepseek→glm, glm→glm) needed to test whether the
    "cross-model semantic misalignment" finding generalizes beyond the
    qwen↔deepseek pair.
    """

    VERIFIER_PROVIDER = "glm"

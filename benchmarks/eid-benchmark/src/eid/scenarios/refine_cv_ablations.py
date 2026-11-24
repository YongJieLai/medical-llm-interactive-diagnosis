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
    """[DEPRECATED] Same-model verification with REFUTATION trigger only.

    BUG: this ablation is structurally deadlocked. REFUTATION fires only when a
    "previous diagnosis" exists, but prev_diagnosis is only updated by the GATHER
    path — which requires gap (or claim) to be enabled. With both disabled, every
    case runs exactly ONE verification pipeline with prev_diagnosis=None, so the
    refutation trigger never fires (verified: 0/60 traces). Use
    RefineCVQwenGapRefuteScenario instead to measure refutation's isolated
    contribution in a multi-round environment.
    """

    ENABLE_CLAIM_CHECK = False
    ENABLE_GAP_CHECK = False


class RefineCVQwenGapRefuteScenario(RefineCVQwenScenario):
    """GAP CHECK + REFUTATION (no claim check).

    This is the correct way to isolate REFUTATION's contribution: gap check
    creates the multi-round environment (prev_diagnosis gets populated), and
    refutation then fires when the diagnosis is unchanged across rounds.
    Contrast with RefineCVQwenGapOnlyScenario (= gap without refutation) to get
    the NET contribution of refutation.
    """

    ENABLE_CLAIM_CHECK = False
    ENABLE_GAP_CHECK = True
    ENABLE_REFUTATION = True


class RefineCVQwenGapRefuteRAGScenario(RefineCVQwenGapRefuteScenario):
    """GAP + REFUTATION + external DDX-KB (knowledge-grounded refutation).

    Same as GapRefute but with ENABLE_RAG=True: the verifier's refutation
    counter-evidence / alternative are grounded in the external differential
    diagnosis knowledge base instead of the verifier's (limited) memory.

    ⚠️ ORACLE KB — the KB disease set == the answer diseases, so results are a
    KNOWLEDGE-CEILING analysis, not a formal RAG method.
    """

    ENABLE_RAG = True


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

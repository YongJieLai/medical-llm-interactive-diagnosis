"""方案 C1：判别性证据导向采集（共性层快速过 + 判别层优先深采）。

背景（`docs/wrong-entity-deepdive.md`）：方案 A 把覆盖率 31%→50%，但采到的大多是
"共性证据"（症状/病程，所有候选病共享），不是"判别证据"（化验/影像/病原学/药物史，
能把同系统候选病切开的客观发现）。剩下 35 例 wrong_entity 缺的正是判别证据。

本方案在方案 A 的 8 维框架上做"证据分层"：把 8 维分成共性层(1-3)与判别层(4-8)，
明确告诉 doctor：共性症状不足以确诊，应尽快转向客观判别证据。
- 纯 prompt 级、零额外 LLM 调用、零训练、不泄题。
- 与 P0 / 方案 B 的本质区别：不要求输出 top-k 假设，只静态标注"哪些证据类别是
  判别性的"，把"问什么具体问题"仍留给 doctor 自己（继承方案 A 的成功前提）。
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario

# 证据分层框架：共性层 vs 判别层。
_EVIDENCE_FRAMEWORK_LAYERED = (
    "\n"
    "Evidence Framework — you must cover ALL eight domains below before issuing [FINISH].\n"
    "The eight domains split into TWO layers with very different diagnostic value:\n"
    "\n"
    "  COMMON-SYMPTOM layer (shared by many candidate diseases — cover quickly, do NOT dwell here):\n"
    "    1) Chief complaint & onset\n"
    "    2) Course of present illness (progression, triggers, timing)\n"
    "    3) Associated symptoms / focused review of systems\n"
    "\n"
    "  DISCRIMINATING layer (objective findings that actually ESTABLISH the diagnosis — prioritize these):\n"
    "    4) Past medical & surgical history\n"
    "    5) Medications & allergies\n"
    "    6) Social / occupational / exposure history\n"
    "    7) Targeted physical examination\n"
    "    8) Most informative laboratory / imaging / microbiology / pathology\n"
    "\n"
    "  Once the COMMON-SYMPTOM layer is covered, do NOT keep asking more symptom details — "
    "move to the DISCRIMINATING layer as soon as possible. The discriminating layer is where "
    "the diagnosis is actually made: a specific lab value, imaging finding, microbe, drug or "
    "exposure separates look-alike diseases that share identical symptoms.\n"
)

_EFFICIENCY_LINE = "1. Efficiency: Gather sufficient information in as few turns as possible."
_COMPREHENSIVE_LINE = (
    "1. Comprehensiveness: Gather a COMPLETE evidence picture across all eight "
    "domains below, prioritizing the DISCRIMINATING layer. Do NOT finish early just to save turns."
)
_TURN_ANCHOR = "In every turn, follow a strict 'Reasoning-then-Acting' process:"


class RefineCVDiscrimScenario(RefineCVScenario):
    """方案 C1：判别性证据导向采集（共性层快过 + 判别层优先）+ 原样四检查门控。"""

    MODE = "refine_cv_discrim"

    def _doctor_system_prompt(self) -> str:
        base = super()._doctor_system_prompt()  # SC evidence-gatherer doctor（含 {max_turns} 占位符）
        # 1) 效率优先 -> 广覆盖 + 判别优先
        base = base.replace(_EFFICIENCY_LINE, _COMPREHENSIVE_LINE)
        # 2) "objective ... efficiently" -> comprehensively（一致性）
        base = base.replace(
            "Your objective is to efficiently gather information and request necessary",
            "Your objective is to comprehensively gather information and request necessary",
        )
        # 3) 在 Reasoning-then-Acting 之前注入分层证据框架
        base = base.replace(
            _TURN_ANCHOR, _EVIDENCE_FRAMEWORK_LAYERED + "\n" + _TURN_ANCHOR
        )
        return base

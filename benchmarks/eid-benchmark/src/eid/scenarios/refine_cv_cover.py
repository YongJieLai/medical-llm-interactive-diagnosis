"""方案 A：广覆盖采集（去「效率优先」→ 结构化证据框架驱动）。

背景（`docs/main-battlefield-cover-driven-collection.md`）：主战场在采集层，
覆盖率中位仅 30%（正确诊断链路的瓶颈）。元凶之一是 doctor system prompt 的
第 1 条 "Efficiency: as few turns as possible"——它在采集期就鼓励"少问、窄问、
快结束"。

本方案最小改动：把"效率优先"换成"广覆盖"，并注入 8 维结构化证据框架作为
"关键证据类别"的候选空间，让 doctor 在 16 轮内系统性地覆盖完整证据面，而不是
围绕当前猜测问窄问少。

- 纯 prompt 级、零额外 LLM 调用、零训练、不泄题。
- 其余四检查门控（claim/gap/refutation/decisive）原样保留，只改采集层。
- 与 P0 / 两阶段的本质区别：本方案锚点是"证据类别缺口"（cover），不是
  "诊断假设对"（discriminate）。
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario

# 8 维结构化证据框架：作为"关键证据类别"的候选空间注入 doctor。
_EVIDENCE_FRAMEWORK = (
    "\n"
    "Evidence Framework — you must systematically cover ALL eight domains below "
    "before issuing [FINISH]. Missing a key domain is worse than using one extra turn:\n"
    "  1) Chief complaint & onset\n"
    "  2) Course of present illness (progression, triggers, timing)\n"
    "  3) Associated symptoms / focused review of systems\n"
    "  4) Past medical & surgical history\n"
    "  5) Medications & allergies\n"
    "  6) Social / occupational / exposure history\n"
    "  7) Targeted physical examination\n"
    "  8) Most informative laboratory / imaging\n"
)

_EFFICIENCY_LINE = "1. Efficiency: Gather sufficient information in as few turns as possible."
_COMPREHENSIVE_LINE = (
    "1. Comprehensiveness: Gather a COMPLETE evidence picture across all eight "
    "domains below. Do NOT finish early just to save turns."
)
_TURN_ANCHOR = "In every turn, follow a strict 'Reasoning-then-Acting' process:"


class RefineCVCoverScenario(RefineCVScenario):
    """广覆盖采集 + 原样四检查门控。"""

    MODE = "refine_cv_cover"

    def _doctor_system_prompt(self) -> str:
        base = super()._doctor_system_prompt()  # SC evidence-gatherer doctor（含 {max_turns} 占位符）
        # 1) 效率优先 -> 广覆盖
        base = base.replace(_EFFICIENCY_LINE, _COMPREHENSIVE_LINE)
        # 2) "objective ... efficiently" -> comprehensively（一致性）
        base = base.replace(
            "Your objective is to efficiently gather information and request necessary",
            "Your objective is to comprehensively gather information and request necessary",
        )
        # 3) 在 Reasoning-then-Acting 之前注入 8 维证据框架
        base = base.replace(
            _TURN_ANCHOR, _EVIDENCE_FRAMEWORK + "\n" + _TURN_ANCHOR
        )
        return base

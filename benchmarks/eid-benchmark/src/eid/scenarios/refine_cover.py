"""消融：无 CEV-GR 的广覆盖采集（refine + 去效率 + 8 维证据框架）。

回答的问题：方案A（refine_cv_cover, 65.0%）的 +6.5pp 里，8 维证据框架的贡献
是否依赖 CEV-GR 四检查门控（claim/gap/refutation/decisive）。

- 本场景继承 clean baseline `RefineCleanScenario`（同模型 verifier、仅 INCOMPLETE
  重采、无 claim/gap/refutation/decisive 四门控），只改 doctor system prompt 的三处
  （去效率优先 + 注入 8 维证据框架），改动与 `refine_cv_cover` 完全一致。
- 与 `refine_cv_cover` 的唯一区别：后者继承 RefineCVScenario（带 CEV-GR），
  本场景继承 RefineCleanScenario（无 CEV-GR）。
- 若 refine_cover ≈ refine_cv_cover，说明 8 维框架独立有效（独立叠加）；
  若 refine_cover ≈ refine（贴着 55.0%），说明 8 维框架依赖 CEV-GR（协同）。
"""
from __future__ import annotations

from eid.scenarios.refine_clean import RefineCleanScenario
from eid.scenarios.refine_cv_cover import (
    _EVIDENCE_FRAMEWORK,
    _EFFICIENCY_LINE,
    _COMPREHENSIVE_LINE,
    _TURN_ANCHOR,
)


class RefineCoverScenario(RefineCleanScenario):
    """广覆盖采集，无 CEV-GR 四门控（消融对照组）。"""

    MODE = "refine_cover"

    def _doctor_system_prompt(self) -> str:
        base = super()._doctor_system_prompt()
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

"""消融对照：只去「效率优先」、不加 8 维证据框架（隔离"多问"效应）。

公平性检查（方案 A vs 基线）发现方案 A 混淆了两个变量：
  变量 A「多问」：去掉 "as few turns as possible"，允许用满 16 轮预算；
  变量 B「问得好」：注入 8 维结构化证据框架，让每轮采集更有方向。

本场景只保留变量 A（去效率优先、鼓励用满预算），不注入框架。用它做
"框架有无"的单一变量消融：

  refine_cv（基线）   : 效率优先 + 无框架
  refine_cv_freebudget: 去效率   + 无框架   -> 隔离"多问"效应
  refine_cv_cover     : 去效率   + 8维框架 -> 方案 A

  cover - freebudget = 框架的净贡献（同停止策略下）
  freebudget - base  = 多问的净效应
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario

_EFFICIENCY_LINE = "1. Efficiency: Gather sufficient information in as few turns as possible."
_FREEBUDGET_LINE = (
    "1. Budget use: You have a full {max_turns}-turn budget. Use it thoroughly "
    "to gather a complete picture; do NOT finish early just to save turns."
)
_TURN_ANCHOR = "In every turn, follow a strict 'Reasoning-then-Acting' process:"


class RefineCVFreebudgetScenario(RefineCVScenario):
    """去效率优先（允许用满预算）+ 原样四检查门控，但不注入证据框架。"""

    MODE = "refine_cv_freebudget"

    def _doctor_system_prompt(self) -> str:
        base = super()._doctor_system_prompt()  # SC evidence-gatherer doctor
        # 只把"效率优先"换成"用满预算"，其它原样（不注入 Evidence Framework）。
        base = base.replace(_EFFICIENCY_LINE, _FREEBUDGET_LINE)
        base = base.replace(
            "Your objective is to efficiently gather information and request necessary",
            "Your objective is to thoroughly gather information and request necessary",
        )
        return base

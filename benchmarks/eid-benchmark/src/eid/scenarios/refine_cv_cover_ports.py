"""方案A（8维框架 + CEV-GR 四检查门控）移植到其它底座：React / Roleplay。

背景（`docs/cev-gr-host-portability.md`）：CEV-GR 四检查门控已经移植过
（react_cv / react_cv2 / roleplay_cv），但当时只移了 CEV-GR，没移方案A的
8 维证据框架（采集层改动）。本文件补上这一环——把完整方案A
（`RefineCVCoverScenario`：去效率 + 8维框架 + 四检查门控）移植到 React /
Roleplay 两个底座，验证 8 维框架在其它底座上是否同样有效。

移植方式与 `refine_cv_ports.py` 一致：只需覆写
  - `_doctor_system_prompt()`  -> 宿主底座的 doctor prompt + 方案A三处采集层改动
  - `_is_finish_action()`      -> 宿主的"结束"动作

注意各底座的 8 维框架注入锚点（"In every turn, ..."）不同，需分别传入。
"""
from __future__ import annotations

from eid.scenarios.refine_cv_cover import RefineCVCoverScenario, _apply_cover_edits

# React doctor 的 turn anchor（"you must follow"）。
_REACT_TURN_ANCHOR = "In every turn, you must follow a strict 'Reasoning-then-Acting' process:"
# Base roleplay doctor 的 turn anchor（无 Reasoning-then-Acting）。
_BASE_TURN_ANCHOR = (
    "In every turn, you execute one of the following actions in the corresponding format:"
)


class ReactCoverScenario(RefineCVCoverScenario):
    """React (ReAct) base + 方案A（8维框架 + CEV-GR 四检查门控）。

    ReAct doctor 的终点动作是 [DIAGNOSIS]（无 [FINISH]），故 finish = diagnosis。
    """

    MODE = "react_cover"

    def _doctor_system_prompt(self) -> str:
        return _apply_cover_edits(
            self.prompts.get_react_doctor_system_prompt(), _REACT_TURN_ANCHOR
        )

    def _is_finish_action(self, action_type: str) -> bool:
        return action_type == "diagnosis"


class RoleplayCoverScenario(RefineCVCoverScenario):
    """Roleplay (base doctor) base + 方案A（8维框架 + CEV-GR 四检查门控）。

    base doctor 的终点动作是 [DIAGNOSIS]（无 [FINISH]），故 finish = diagnosis。
    """

    MODE = "roleplay_cover"

    def _doctor_system_prompt(self) -> str:
        return _apply_cover_edits(
            self.prompts.get_base_doctor_system_prompt(), _BASE_TURN_ANCHOR
        )

    def _is_finish_action(self, action_type: str) -> bool:
        return action_type == "diagnosis"

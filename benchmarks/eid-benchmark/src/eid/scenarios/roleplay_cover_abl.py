"""消融：roleplay 底座 + 8 维证据框架（无 CEV-GR 四检查门控）。

补齐跨底座消融矩阵缺失的一格。背景见 `docs/plugin-portability-cover.md`：

  refine 底座四格已闭合：
    refine          55.0%（无 CEV-GR baseline）
    refine_cv       58.5%（+CEV-GR）
    refine_cover    60.0%（+8维框架，无 CEV-GR）
    refine_cv_cover 65.0%（完整插件）

  roleplay 底座缺"8维框架独立贡献"这一格：
    roleplay             49.0%（无 CEV-GR baseline）
    roleplay_cv          57.5%（+CEV-GR）
    roleplay_cover_abl   ?     （+8维框架，无 CEV-GR）  ← 本文件补齐
    roleplay_cover       64.0%（完整插件）

  回答的问题：8 维框架在 roleplay 底座上是否像 refine 底座一样有独立 +5.0pp？
  （roleplay 基线无验证环，医生给 [DIAGNOSIS] 直接结束，不能假设与 refine 等价。）

实现：继承无 CEV-GR 的 `RoleplayScenario`，只对 doctor prompt 应用方案A 三处
采集层改动（去效率 + 8维框架），与 `refine_cover`（refine 底座对应物）同构。
"""
from __future__ import annotations

from eid.scenarios.roleplay import RoleplayScenario
from eid.scenarios.refine_cv_cover import _apply_cover_edits

# base roleplay doctor 的 turn anchor（无 Reasoning-then-Acting）。
_BASE_TURN_ANCHOR = (
    "In every turn, you execute one of the following actions in the corresponding format:"
)


class RoleplayCoverAblScenario(RoleplayScenario):
    """roleplay + 8 维框架（无 CEV-GR），隔离 8 维框架在 roleplay 上的独立贡献。"""

    MODE = "roleplay_cover_abl"

    def _doctor_system_prompt(self) -> str:
        return _apply_cover_edits(
            self.prompts.get_base_doctor_system_prompt(), _BASE_TURN_ANCHOR
        )

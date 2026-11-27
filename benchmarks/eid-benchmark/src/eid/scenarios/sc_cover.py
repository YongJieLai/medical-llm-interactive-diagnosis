"""SC 底座 + 方案A 的 8 维框架采集层改动（不含 CEV-GR 四检查门控）。

背景：SC 的 doctor prompt 与 refine/refine_cv 共用同一个
`get_sc_doctor_system_prompt`，但诊断管线不同——SC 是「summarizer ->
diagnostician」一步诊断（无验证环），refine 是「diagnostician -> verifier」
验证环。

本文件在 SC 原版管线（summarizer + diagnostician，不加 CEV-GR）上，只对采集层
应用方案A的三处改动（去效率 + 8维框架），用来回答：
  「8 维框架的采集增益在 SC 的 summarizer 诊断管线上是否也成立」
  对照 refine_cover（8 维框架在 refine 验证环管线上，+5.0pp）。
"""
from __future__ import annotations

from eid.scenarios.sc import SCScenario
from eid.scenarios.refine_cv_cover import _apply_cover_edits

# SC doctor 的 turn anchor（与 refine_cv_cover 一致）。
_SC_TURN_ANCHOR = "In every turn, follow a strict 'Reasoning-then-Acting' process:"


class SCCoverScenario(SCScenario):
    """SC 底座 + 8 维框架采集层改动（无 CEV-GR，保留 summarizer 诊断管线）。"""

    MODE = "sc_cover"

    def _doctor_system_prompt(self) -> str:
        return _apply_cover_edits(
            self.prompts.get_sc_doctor_system_prompt(), _SC_TURN_ANCHOR
        )

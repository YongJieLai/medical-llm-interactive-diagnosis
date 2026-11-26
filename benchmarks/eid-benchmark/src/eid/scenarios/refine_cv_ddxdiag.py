"""P4 决策防过度排除（方案2）：两步法前移到下游决策组件。

根因（`docs/p3-doctor-ddx.md`）：P3 doctorddx 只 +3.3pp，因为「证据采集不足」实为
「采集不足 + 下游过度排除」两环节。采集侧已修（doctorddx），下游两个组件仍过度排除：
  - diagnostician：用经验规则过早排除（medqa_25 PSC 被「focal→恶性肿瘤」）
  - summarizer：诊断性裁剪丢关键证据（medqa_9 采到 fever 却排除 endometritis）

三路并行消融（同口径：qwen-plus / 10 turns / 前 30 例 / 确定性 judge）：
  - 2a diagnostician 两步法：强制先枚举候选，排除必须有直接反证（治 35% 主战场）
  - 2b summarizer 中性转录：忠实保留全部阳性/关键阴性发现，不做诊断性裁剪
  - 2c 联合：2a + 2b

不泄题：纯 prompt 改动，零训练，不注入任何答案信息。
"""
from __future__ import annotations

from eid.scenarios.refine_cv_ablations import RefineCVQwenGapRefuteScenario


class RefineCVQwenGapRefuteDDXDiagScenario(RefineCVQwenGapRefuteScenario):
    """2a：diagnostician 两步法（显式枚举候选 + 排除需直接反证）。"""

    def _diagnostician_system_prompt(self) -> str:
        return self.prompts.get_diagnostician_cv_ddx_system_prompt()


class RefineCVQwenGapRefuteNeutralSummScenario(RefineCVQwenGapRefuteScenario):
    """2b：summarizer 中性转录（不诊断性裁剪）。"""

    def _summarizer_system_prompt(self) -> str:
        return self.prompts.get_summarizer_neutral_system_prompt()


class RefineCVQwenGapRefuteJointScenario(RefineCVQwenGapRefuteScenario):
    """2c：2a + 2b 联合。"""

    def _summarizer_system_prompt(self) -> str:
        return self.prompts.get_summarizer_neutral_system_prompt()

    def _diagnostician_system_prompt(self) -> str:
        return self.prompts.get_diagnostician_cv_ddx_system_prompt()

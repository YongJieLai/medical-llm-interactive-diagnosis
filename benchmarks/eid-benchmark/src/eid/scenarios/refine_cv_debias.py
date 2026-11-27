"""诊断层去偏置（方案 1：诊断分级强制）——治 overdiagnosis，改门槛不改倾向。

背景：
- `docs/error-distribution-and-improvement.md`：overdiagnosis 占错题 14–16%，且其
  覆盖率不低（32–33%）——证据采到了一些，但诊断器把常见/良性/功能性疾病升级成
  严重/罕见/器质性疾病。
- `docs/p0p1-improvement-results.md`：P1 用 base-rate prior（"优先常见病"）治它，
  结果只是把 overdiagnosis（12→3 例）换成 wrong_entity（29→35 例），准确率不变——
  因为加先验本质是"用一个新 bias 抵消旧 bias"，当 ground truth 真的是严重病时，
  就被拉向更常见的错误答案。

方案 1 与 P1 的本质区别：
- P1 改的是「倾向」（prior 偏向常见病）；
- 方案 1 改的是「输出结构 + 举证门槛」：要求 diagnostician 先列出"想升级但证据
  不足"的候选（[ESCALATION]，明确不计入答案），再在 [DIAGNOSIS] 里只给"证据能
  直接确立的最高诊断层级"。升级从"默认允许"变成"需特定阳性证据举证"。

关键实现点：把 [ESCALATION] 放在 [DIAGNOSIS] 之前、[DIAGNOSIS] 保持为最后一段，
使 BaseScenario.extract_action 用 `[DIAGNOSIS]\s*([\s\S]*)` 提取的最终答案天然不
包含升级候选（升级候选在其之前），无需改动判分链路。

纯 prompt 改动、零额外 LLM 调用、零训练、不泄题。
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario
from eid.scenarios.refine_cv_cover import RefineCVCoverScenario

# ------------------------------------------------------------------ #
# 替换片段（模块级，供基线版 / 方案A叠加版复用）
# ------------------------------------------------------------------ #
# 锚点：diagnostician cv system prompt 里 [DIAGNOSIS] 段开头，唯一出现。
_DEBIAS_OLD = "[DIAGNOSIS]\n   - Provide the "

_DEBIAS_NEW = (
    "[ESCALATION]\n"
    "   - List any diagnosis that is MORE severe, rarer, or higher-acuity than your "
    "final answer ONLY if you are tempted to make it — and for each, name the SPECIFIC "
    "supporting finding it would require that is NOT yet present in the summary.\n"
    "   - This section is UNVERIFIED suspicion. It is NOT your answer and must NOT be "
    "repeated in [DIAGNOSIS].\n"
    "   - If no escalation is warranted, write 'None'.\n"
    "\n"
    "[DIAGNOSIS]\n"
    "   - State the HIGHEST diagnostic level that the collected evidence DIRECTLY "
    "establishes (what you can CONFIRM now), not the most severe disease you can imagine. "
    "A common, benign, or functional diagnosis that fits all findings is a VALID final "
    "answer; do NOT upgrade to a severe or rare disease unless a SPECIFIC reported finding "
    "directly supports it.\n"
    "   - Provide the "
)


def _debias_prompt(base: str) -> str:
    """把 [DIAGNOSIS] 段替换为 [ESCALATION] + 分级强制的 [DIAGNOSIS]。"""
    return base.replace(_DEBIAS_OLD, _DEBIAS_NEW) if _DEBIAS_OLD in base else base


class RefineCVDebiasScenario(RefineCVScenario):
    """方案 1：基线 refine_cv + 诊断分级强制（治 overdiagnosis）。"""

    MODE = "refine_cv_debias"

    def _diagnostician_system_prompt(self) -> str:
        return _debias_prompt(super()._diagnostician_system_prompt())


class RefineCVCoverDebiasScenario(RefineCVCoverScenario):
    """方案 A（广覆盖采集）+ 诊断分级强制：完整方法（采集层 + 诊断层）。"""

    MODE = "refine_cv_cover_debias"

    def _diagnostician_system_prompt(self) -> str:
        return _debias_prompt(super()._diagnostician_system_prompt())

"""P0 + P1 最小改动：采集层「动态 top-2 区分缺口驱动」+ 诊断层「base-rate prior」。

背景（`docs/error-distribution-and-improvement.md`）：refine_cv 的错误分布里约 70% 是
「离题(off_system) + 同系统错实体(wrong_entity)」，二者覆盖率仅 28–32%（采集不足）；
另有 14–16% 是「过度诊断」，其覆盖率不低（32–33%），是诊断器把常见病升级成严重病
的系统性偏置。

P0（采集层，主杠杆）：把 doctor 的每轮 [THOUGHT] 从「分析画面找信息缺口」（被动）
改为「维护 top-2 鉴别诊断 → 找出区分二者的单一最判别性缺口 → 针对性提问」（主动）。
    纯 prompt 改动、零额外 LLM 调用、零训练、不泄题。
    相比 P3 doctorddx 的「静态采集清单」（生成一次、跨轮复用、要求系统覆盖 4–6 候选，
    实测仅 +3.3pp），本方案是「动态、聚焦」的：每轮只追 top-2 的判别缺口，不摊薄火力。

P1（诊断层，次杠杆）：给 diagnostician 加 base-rate prior（Occam's razor），
    「优先常见/简单/功能性/自限性诊断，除非有明确报告的器质性阳性证据」，
    对冲 14–16% 的过度诊断。纯 prompt 改动。

两组件均独立、可叠加，均基于主推的 RefineCVScenario（deepseek 跨模型验证底座）。
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario

# ------------------------------------------------------------------ #
# 替换片段（模块级，供 P0 / P1 / P0P1 复用）
# ------------------------------------------------------------------ #
_P0_OLD = (
    "[THOUGHT] <Your Clinical Reasoning>\n"
    "   - Analyze the current clinical picture and identify critical information gaps.\n"
)
_P0_NEW = (
    "[THOUGHT] <Your Clinical Reasoning>\n"
    "   - Maintain a running differential: state your current top-2 candidate "
    "diagnoses (after turn 1, once a chief complaint exists; if the data is still "
    "too sparse, state the single most likely direction instead).\n"
    "   - Identify the SINGLE most discriminating question or test that best "
    "separates those top-2 candidates.\n"
    "   - Your next action must target exactly that discriminating gap.\n"
)

_P1_OLD = (
    "[THOUGHT] <Your Clinical Reasoning>\n"
    "   - Perform a comprehensive clinical analysis of the summary.\n"
)
_P1_NEW = (
    "[THOUGHT] <Your Clinical Reasoning>\n"
    "   - Perform a comprehensive clinical analysis of the summary.\n"
    "   - Apply a base-rate prior: among all diagnoses consistent with the evidence, "
    "prefer the MOST COMMON / SIMPLEST / FUNCTIONAL / SELF-LIMITED one. Only escalate "
    "to a rarer or more severe organic disease when a SPECIFIC reported finding "
    "directly supports it.\n"
)


def _p0_doctor_prompt(base: str) -> str:
    return base.replace(_P0_OLD, _P0_NEW) if _P0_OLD in base else base


def _p1_diagnostician_prompt(base: str) -> str:
    return base.replace(_P1_OLD, _P1_NEW) if _P1_OLD in base else base


class RefineCVP0Scenario(RefineCVScenario):
    """P0: 动态 top-2 区分缺口驱动提问（治 off_system + wrong_entity，约 70% 错题）。"""

    MODE = "refine_cv_p0"

    def _doctor_system_prompt(self) -> str:
        return _p0_doctor_prompt(super()._doctor_system_prompt())


class RefineCVP1Scenario(RefineCVScenario):
    """P1: diagnostician base-rate prior（治 overdiagnosis，约 14–16% 错题）。"""

    MODE = "refine_cv_p1"

    def _diagnostician_system_prompt(self) -> str:
        return _p1_diagnostician_prompt(super()._diagnostician_system_prompt())


class RefineCVP0P1Scenario(RefineCVScenario):
    """P0 + P1 联合：采集层 DDx 缺口驱动 + 诊断层 base-rate prior。"""

    MODE = "refine_cv_p0p1"

    def _doctor_system_prompt(self) -> str:
        return _p0_doctor_prompt(super()._doctor_system_prompt())

    def _diagnostician_system_prompt(self) -> str:
        return _p1_diagnostician_prompt(super()._diagnostician_system_prompt())

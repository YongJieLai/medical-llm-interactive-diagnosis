"""两阶段采集：冷启动被动探索 → 关键节点主动区分（阈值触发版）。

背景（`docs/p0p1-improvement-results.md`）：P0（全程 top-2 区分提问）实测 −4.5pp，
根因是「在证据稀疏时锁错 top-2 方向，后续区分性提问在错误方向内打转，漏掉
纠正性关键证据」。

本方案（用户提案）：**前期不主动选问题**（被动广覆盖采集，避免过早锁死方向），
等采集到足够信息、到达「关键节点」后再切换到主动区分提问（top-2 + 区分性缺口）。

触发条件（本版）：**轮次阈值 TURN_THRESHOLD** —— 前 K 个问题被动探索，之后主动区分。
它是「信息量」的最简代理；后续可升级为「假设稳定性触发」（连续 N 轮 top-2 不变）。

实现：纯 prompt 级、零额外 LLM 调用、零训练、不泄题。
- 阶段 1（被动）：doctor 保持基线自由采集，但 instruction 明确「广覆盖、别锚定单一假设」。
- 阶段 2（主动）：instruction 切换为「维护 top-2 → 找区分性缺口 → 针对性提问」。
切换由 `_decorate_doctor_instruction` 内部计数器驱动（每 case 新建 scenario 对象，
计数器天然按 case 归零）。
"""
from __future__ import annotations

from eid.scenarios.refine_cv import RefineCVScenario

# ------------------------------------------------------------------ #
# 两阶段的每轮 instruction 追加片段
# ------------------------------------------------------------------ #
_STAGE1_SUFFIX = (
    "\n\n[COLLECTION PHASE -- do not commit to a diagnosis yet]\n"
    "You are in the early information-gathering phase. Keep an OPEN differential and "
    "do NOT anchor on a single hypothesis. Systematically broaden the picture: chief "
    "complaint, onset and course of the present illness, associated symptoms, pertinent "
    "past medical / medication / social history, and the most informative basic exam or "
    "labs. Ask ONE open, high-yield question that WIDENS your evidence base, rather than "
    "one that only confirms a premature guess.\n"
)

_STAGE2_SUFFIX = (
    "\n\n[DISCRIMINATING PHASE -- evidence is now sufficient to focus]\n"
    "You have gathered enough evidence to narrow down. State your current top-2 candidate "
    "diagnoses, then identify the SINGLE most discriminating question or test that best "
    "separates those two. Your next action must target exactly that discriminating gap.\n"
)


class RefineCVStagedScenario(RefineCVScenario):
    """两阶段采集：前 TURN_THRESHOLD 问被动探索，之后主动 top-2 区分。"""

    MODE = "refine_cv_staged"
    TURN_THRESHOLD = 6  # 前 6 个问题被动，之后主动（子类覆盖以扫阈值）

    def _decorate_doctor_instruction(self, instruction: str) -> str:
        # 每 case 新建 scenario 对象，_doctor_turn 天然从 0 起；getattr 仅作兜底。
        t = getattr(self, "_doctor_turn", 0)
        self._doctor_turn = t + 1
        if t >= self.TURN_THRESHOLD:
            return instruction + _STAGE2_SUFFIX
        return instruction + _STAGE1_SUFFIX


class RefineCVStagedK4Scenario(RefineCVStagedScenario):
    """阈值扫描：K=4。"""

    MODE = "refine_cv_staged_k4"
    TURN_THRESHOLD = 4


class RefineCVStagedK8Scenario(RefineCVStagedScenario):
    """阈值扫描：K=8。"""

    MODE = "refine_cv_staged_k8"
    TURN_THRESHOLD = 8

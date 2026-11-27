"""方案 B：证据类别缺口驱动的采集规划（方案 A + 前移 gap check）。

在方案 A（去效率优先 + 8 维证据框架）之上，新增一个采集规划器（复用 DeepSeek
跨模型，与 doctor 解耦）：每轮 doctor-patient/reporter 交换后，规划器读当前对话，
判断 8 个证据域里还有哪些"未覆盖/覆盖不足"，输出定向采集建议；下一轮 doctor 的
per-turn instruction 里注入这份缺口清单，实现"采集期前移 gap check"。

与 P0 / 两阶段 / 现有 gap check 的本质区别：
  - 锚点是「证据类别缺口」（哪个 domain 还没问到），不是「区分 top-2 假设的检查」。
  - category 被约束在 8 维框架内（白名单校验），防止规划器退化成"想到什么问什么"。

对照设计（公平）：
  方案 A（refine_cv_cover）      : 静态 8 维框架，无动态规划
  方案 A+B（refine_cv_cover_plan）: 静态框架 + 每轮动态缺口规划
  两者同底座、同停止策略、同判分口径，唯一变量是"动态缺口规划器"。

成本：每轮最多 +1 次 DeepSeek 调用（轻量，json_mode，输出 ≤512 token）。
"""
from __future__ import annotations

from eid.scenarios.refine_cv_cover import RefineCVCoverScenario

try:
    from llm_client import chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import chat as _llm_chat

# 8 维证据域（下划线形式，与规划器 prompt 的 category 白名单一致）。
_DOMAINS = {
    "chief_complaint",
    "present_illness",
    "associated_symptoms",
    "past_history",
    "medications_allergies",
    "social_occupational_exposure",
    "physical_exam",
    "lab_imaging",
}

_PLANNER_SYSTEM = (
    "You are a clinical evidence-gathering planner for an interactive diagnostic "
    "conversation. Given the conversation so far, identify which of the 8 evidence "
    "domains are still UNCOVERED or under-covered, and propose the single most "
    "diagnostically valuable question for the doctor's NEXT turn.\n"
    "Rules:\n"
    "- category MUST be exactly one of the 8 domains below (lowercase, underscore).\n"
    "- Only propose a question if that domain is genuinely still missing or weak; "
    "do NOT re-ask what is already answered.\n"
    "- Prioritise questions that would ESTABLISH or REFUTE a diagnosis, NOT questions "
    "that merely discriminate between two candidate hypotheses.\n"
    "- Keep each question concrete and answerable by the patient or a test.\n"
    "Evidence domains:\n"
    "  1) chief_complaint\n"
    "  2) present_illness\n"
    "  3) associated_symptoms\n"
    "  4) past_history\n"
    "  5) medications_allergies\n"
    "  6) social_occupational_exposure\n"
    "  7) physical_exam\n"
    "  8) lab_imaging\n"
    'Output JSON only: {"gaps": [{"category": "...", "question": "...", '
    '"why": "one short clause"}]}'
)


class RefineCVCoverPlanScenario(RefineCVCoverScenario):
    """方案 A + 采集规划器（前移 gap check）。"""

    MODE = "refine_cv_cover_plan"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 复用 verifier 的 DeepSeek client/model 作为采集规划器（跨模型解耦）。
        self._current_plan = ""  # 下一轮 doctor instruction 要注入的缺口清单

    # ------------------------------------------------------------------ #
    # 采集规划器
    # ------------------------------------------------------------------ #
    def _plan_gaps(self, dialogue_history: str) -> str:
        """读当前对话，输出"未覆盖证据域"的定向采集建议文本。"""
        ctx = dialogue_history[-6000:] if len(dialogue_history) > 6000 else dialogue_history
        try:
            resp = _llm_chat(
                self._ds_client,
                self._ds_model,
                [
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user", "content": f"Conversation so far:\n{ctx}"},
                ],
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )
        except Exception:  # pragma: no cover - fail open
            return ""
        if not isinstance(resp, dict):
            return ""
        gaps = resp.get("gaps", []) or []
        lines = ["Evidence-gap plan for your NEXT turn (fill the UNCOVERED domains):"]
        n = 0
        for g in gaps:
            if not isinstance(g, dict):
                continue
            cat = str(g.get("category", "")).strip().lower()
            q = str(g.get("question", "")).strip()
            why = str(g.get("why", "")).strip()
            # 白名单校验：category 必须落在 8 维框架内，防止规划器自由退化。
            if cat not in _DOMAINS or not q:
                continue
            suffix = f" ({why})" if why else ""
            lines.append(f"- [{cat}] {q}{suffix}")
            n += 1
            if n >= 3:
                break
        if n == 0:
            return ""
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 两个 hook：每轮交换后规划 → 下一轮 instruction 注入
    # ------------------------------------------------------------------ #
    def _after_dialogue_turn(self, current_turn: int, dialogue_history: str, last_reply: str) -> None:
        # 接近预算上限不再规划（最后 1 轮就要 FINISH，规划无意义）。
        if current_turn >= self.max_turns - 1:
            self._current_plan = ""
            return
        self._current_plan = self._plan_gaps(dialogue_history)

    def _decorate_doctor_instruction(self, instruction: str) -> str:
        if self._current_plan:
            return instruction + "\n\n" + self._current_plan
        return instruction

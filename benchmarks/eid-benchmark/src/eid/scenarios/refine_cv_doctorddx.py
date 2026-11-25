"""P3 证据采集修正（DDx 驱动问诊）：把「两步法」前移到 doctor 采集环节。

根因（`docs/fromhead-reaudit.md`）：26 个真错误里，46%（12 个）是「证据采集不足」——
doctor 问诊锚定单一假设深挖，系统性漏掉正确诊断的前置信息（免疫状态、感染史、
吸烟史、性交痛、用药史、孕期感染、产后感染等）。

机制：doctor 第 1 轮拿到主诉后，用 LLM（DeepSeek，与 verifier 同模型）生成一份
「鉴别诊断采集清单」（top-k 候选 + 每个候选的关键采集点），随后注入 doctor 的
每一轮问诊指令，强制 doctor 系统覆盖多个候选的区分信息，而非锚定单一假设。

与 P2 层 2（verifier 两步法）对称：
  - 层 2 verifier：反证前先展开候选病的鉴别诊断 → +16.7pp
  - 本组件 doctor：问诊前先展开主诉的鉴别诊断 → 治证据采集不足（46%）

不泄题：采集清单由 LLM 基于当前对话（不含 ground truth）现场生成，候选由主诉
自然推导，属于通用问诊策略，不注入任何答案信息。
"""
from __future__ import annotations

import json

from eid.scenarios.refine_cv_ablations import RefineCVQwenGapRefuteScenario

try:
    from llm_client import chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import chat as _llm_chat

_GATHER_SYS = (
    "You are a senior clinician guiding differential-diagnosis-driven history taking. "
    "Based on the chief complaint and information gathered so far, produce a "
    "systematic information-gathering plan. Output JSON ONLY."
)

_GATHER_USER = (
    "Dialogue so far:\n{dialogue}\n\n"
    "Return a JSON object with exactly one key:\n"
    '{{"differential": [{{"candidate": "...", "key_info_to_gather": "..."}}]}}\n'
    "Rules:\n"
    "- List 4-6 candidate conditions that could explain the chief complaint.\n"
    "- For EACH candidate, `key_info_to_gather` must name the SPECIFIC, "
    "discriminating information that would confirm or rule it out (e.g. immune "
    "status for opportunistic infection, medication/substance history, smoking "
    "history, recent infection, sexual/reproductive history, pregnancy/postpartum "
    "status, weight change).\n"
    "- Cover commonly-missed domains; do NOT assume the most obvious diagnosis is "
    "correct.\n"
    "- Be clinically accurate; do NOT invent implausible conditions.\n"
    "Output JSON only, no markdown fences."
)


class RefineCVQwenGapRefuteDoctorDDXScenario(RefineCVQwenGapRefuteScenario):
    """GAP + REFUTATION + DDx-driven history taking (P3 evidence-gathering fix)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gather_plan: str | None = None  # 采集清单（生成一次，跨轮复用）

    # ------------------------------------------------------------------ #
    # Hook 1: 第 1 轮 dialogue 后生成采集清单
    # ------------------------------------------------------------------ #
    def _after_dialogue_turn(
        self, current_turn: int, dialogue_history: str, last_reply: str
    ) -> None:
        # 只在采集到主诉（第 1 轮 patient 回答）后生成一次。
        if self._gather_plan is not None:
            return
        if current_turn < 1:
            return
        self._gather_plan = self._generate_gather_plan(dialogue_history)

    def _generate_gather_plan(self, dialogue: str) -> str:
        """让 LLM 生成「top-k 鉴别诊断 + 每个候选的关键采集点」；失败返回空串。"""
        try:
            resp = _llm_chat(
                self._ds_client,
                self._ds_model,
                [
                    {"role": "system", "content": _GATHER_SYS},
                    {"role": "user", "content": _GATHER_USER.format(dialogue=dialogue)},
                ],
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception:  # pragma: no cover - fail open (退化为 baseline 问诊)
            return ""
        if not isinstance(resp, dict):
            return ""
        return self._format_gather_plan(resp)

    @staticmethod
    def _format_gather_plan(data: dict) -> str:
        ddx = data.get("differential") or []
        if not isinstance(ddx, list) or not ddx:
            return ""
        lines = ["Differential-diagnosis gathering plan:"]
        for item in ddx:
            if isinstance(item, dict) and item.get("candidate"):
                info = item.get("key_info_to_gather", "")
                lines.append(f"- {item['candidate']}: gather {info}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Hook 2: 把采集清单注入 doctor 每轮问诊指令
    # ------------------------------------------------------------------ #
    def _decorate_doctor_instruction(self, instruction: str) -> str:
        if not self._gather_plan:
            return instruction
        return (
            instruction
            + "\n\n### Differential-Diagnosis Gathering Plan (authoritative) ###\n"
            + self._gather_plan
            + "\nSystematically gather the discriminating information for EACH "
            "candidate above before narrowing down. Do NOT anchor on a single "
            "hypothesis and only pursue it."
        )

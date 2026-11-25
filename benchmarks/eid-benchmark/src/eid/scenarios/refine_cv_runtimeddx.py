"""P2 不泄题版（层 2）：运行时 LLM 生成候选病的鉴别诊断 + 区分点。

与 oracle 版 `RefineCVQwenGapRefuteRAGScenario` 的唯一区别是知识来源：
  - oracle : 预存 `ddx_knowledge.json`（答案病子集，泄题）
  - 本版   : 运行时让 LLM 现场展开「候选诊断」的鉴别诊断知识（通用医学知识，
             候选病由 agent 自己诊断提出，不泄题）

等价于「医生翻教科书查当前候选病的鉴别」，而非「查标准答案」。这直接对标
oracle +6.7pp 的增益来源（differential_diagnoses + distinguishing_feature），
只是把「预存」换成「运行时生成」，隔离「两步法（先展开鉴别诊断再反证）」
vs「一步法（verifier 直接反证）」的差异。

机制：_retrieve_ddx(diagnosis) 改为调用 LLM（verifier 同模型 DeepSeek）生成
与 oracle 同构的 snippet，再按原样注入 verifier。带缓存，避免同候选重复生成。
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

_GEN_SYS = (
    "You are a clinical differential-diagnosis knowledge module. For the candidate "
    "diagnosis given, output its differential-diagnosis knowledge as JSON ONLY."
)

_GEN_USER = (
    "Candidate diagnosis: {diagnosis}\n\n"
    "Return a JSON object with exactly these keys:\n"
    '{{"key_features": ["..."], "differential_diagnoses": [{{"name": "...", '
    '"distinguishing_feature": "..."}}], "pitfalls": ["..."]}}\n'
    "Rules:\n"
    "- key_features: 3-6 typical clinical features of the candidate diagnosis.\n"
    "- differential_diagnoses: 3-6 conditions commonly confused with it, each with "
    "a concrete, specific distinguishing feature.\n"
    "- pitfalls: 1-3 common diagnostic errors.\n"
    "- Be clinically accurate; do NOT invent rare or implausible conditions.\n"
    "Output JSON only, no markdown fences."
)


class RefineCVQwenGapRefuteLLMDDXScenario(RefineCVQwenGapRefuteScenario):
    """GAP + REFUTATION + 运行时 LLM 生成鉴别诊断知识（不泄题，P2 层 2）。"""

    ENABLE_RAG = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._llm_ddx_cache: dict[str, str] = {}

    def _retrieve_ddx(self, diagnosis: str) -> str:
        """运行时生成候选诊断的鉴别诊断 snippet（带缓存）。"""
        d = self._norm_med(diagnosis)
        if not d:
            return ""
        if d in self._llm_ddx_cache:
            return self._llm_ddx_cache[d]
        snippet = self._generate_ddx(diagnosis)
        self._llm_ddx_cache[d] = snippet
        return snippet

    def _generate_ddx(self, diagnosis: str) -> str:
        """调用 LLM 生成与 oracle 同构的 snippet；失败返回空串（fail-open）。"""
        try:
            resp = _llm_chat(
                self._ds_client,
                self._ds_model,
                [
                    {"role": "system", "content": _GEN_SYS},
                    {"role": "user", "content": _GEN_USER.format(diagnosis=diagnosis)},
                ],
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception:  # pragma: no cover - fail open
            return ""
        if not isinstance(resp, dict):
            return ""
        return self._format_snippet(resp)

    @staticmethod
    def _format_snippet(data: dict) -> str:
        """把生成 JSON 格式化成与 oracle _retrieve_ddx 一致的 snippet 文本。"""
        lines: list[str] = []
        ddx = data.get("differential_diagnoses") or []
        if isinstance(ddx, list) and ddx:
            lines.append("Differential diagnoses:")
            for item in ddx:
                if isinstance(item, dict) and item.get("name"):
                    feat = item.get("distinguishing_feature", "")
                    lines.append(f"- {item['name']}: {feat}")
        kf = data.get("key_features") or []
        if kf:
            lines.append("Key features: " + "; ".join(str(x) for x in kf))
        pit = data.get("pitfalls") or []
        if pit:
            lines.append("Common pitfalls: " + "; ".join(str(x) for x in pit))
        return "\n".join(lines)

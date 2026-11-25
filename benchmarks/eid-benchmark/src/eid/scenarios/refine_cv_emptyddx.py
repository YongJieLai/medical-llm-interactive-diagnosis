"""实验③方案 1：空 DDx 占位对照。

与层 2 `RefineCVQwenGapRefuteLLMDDXScenario` 的唯一区别是知识来源：
  - 层 2 : 运行时让 LLM 现场生成「候选诊断」的鉴别诊断 + 区分点（真实内容）
  - 本版 : _retrieve_ddx 返回「结构相同、内容为空」的占位文本，不调用任何 LLM

目的：把层 2 的 +16.7pp 增益，拆成「注入动作 / 两步法格式」vs「知识内容」两个来源。
  - 若本版也有明显增益 → 增益来自「注入动作/格式」，与任何知识内容（含 MedQA 记忆）
    无关，污染担忧排除（空占位里没有任何记忆）。
  - 若本版增益消失、只有层 2 有 → 增益绑定在「知识内容」上，需进一步追「通用医学知识
    vs MedQA 记忆」（进入方案 2 PrimeKG 或方案 3 反事实 benchmark）。

注入路径与 oracle/层 2 完全一致：ENABLE_RAG=True，_verify 里 user_prompt += _rag_inject(
_retrieve_ddx(current_diagnosis))，_rag_inject 用默认（oracle）framing。唯一变量是
snippet 的内容是否为空。
"""
from __future__ import annotations

from eid.scenarios.refine_cv_ablations import RefineCVQwenGapRefuteScenario

# 结构对齐层 2 _format_snippet 的三大段（Differential diagnoses / Key features /
# Common pitfalls），但每个字段内容为空占位。用 "(none available)" 而非空串，
# 这样注入动作真实发生（ENABLE_RAG 分支确实 append 了文本），只是没有知识内容。
_EMPTY_SNIPPET = (
    "Differential diagnoses:\n"
    "- (none available)\n\n"
    "Key features: (none available)\n\n"
    "Common pitfalls: (none available)"
)


class RefineCVQwenGapRefuteEmptyDDXScenario(RefineCVQwenGapRefuteScenario):
    """GAP + REFUTATION + 空 DDx 占位（不调用任何 LLM，不注入任何知识内容）。"""

    ENABLE_RAG = True

    def _retrieve_ddx(self, diagnosis: str) -> str:
        """返回结构相同、内容为空的占位 snippet（覆盖父类的 oracle 查表）。"""
        if not diagnosis:
            return ""
        return _EMPTY_SNIPPET

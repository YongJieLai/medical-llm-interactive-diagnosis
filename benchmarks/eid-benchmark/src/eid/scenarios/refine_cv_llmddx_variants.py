"""层 2（运行时 LLM DDx）的组合与模型变体。

在 `RefineCVQwenGapRefuteLLMDDXScenario`（层 2：verifier 两步法 + 运行时 DDx 生成）
基础上做两个变体：

1. RefineCVQwenGapRefuteLLMDDXNeutralSummScenario —— 层 2 + 2b
   verifier 两步法 + 运行时 DDx 生成（DeepSeek）+ summarizer 中性转录。
   目标：判断「下游信息完整性（2b）」与「上游反证/补采（层2）」是否叠加。

2. RefineCVQwenGapRefuteLLMDDXQwenGenScenario —— 层 2 DDx 生成换 qwen-plus
   只把 `_generate_ddx` 的生成模型从 deepseek-chat 换成 qwen-plus，
   verifier 仍用 DeepSeek（保持不变）。
   目标：隔离「知识内容增益（+3.3pp）是否来自 DeepSeek 强模型」这一混杂变量。
     - 仍 ≈ +16.7pp → 知识内容不依赖生成模型强弱，机制鲁棒；
     - 掉到 ≈ +13.3pp（空占位水平）→ 那 3.3pp 主要靠 DeepSeek 强模型；
     - 掉到 < +13.3pp → qwen-plus 生成的 DDx 引入噪声，反噬两步法。

不泄题：纯 prompt / 生成模型切换，零训练，不注入任何答案信息。
"""
from __future__ import annotations

from eid.scenarios.refine_cv_runtimeddx import (
    RefineCVQwenGapRefuteLLMDDXScenario,
    _GEN_SYS,
    _GEN_USER,
)

try:
    from llm_client import get_llm, chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import get_llm, chat as _llm_chat


class RefineCVQwenGapRefuteLLMDDXNeutralSummScenario(
    RefineCVQwenGapRefuteLLMDDXScenario
):
    """层 2 + 2b：verifier 两步法 + 运行时 DDx 生成 + summarizer 中性转录。"""

    def _summarizer_system_prompt(self) -> str:
        return self.prompts.get_summarizer_neutral_system_prompt()


class RefineCVQwenGapRefuteLLMDDXQwenGenScenario(
    RefineCVQwenGapRefuteLLMDDXScenario
):
    """层 2 的 DDx 知识生成换 qwen-plus（verifier 仍 DeepSeek）。

    与父类唯一区别：`_generate_ddx` 用 qwen-plus client/model 而非 DeepSeek。
    """

    DDX_GEN_PROVIDER = "qwen"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ddx_gen_client, self._ddx_gen_model = get_llm(self.DDX_GEN_PROVIDER)

    def _generate_ddx(self, diagnosis: str) -> str:
        """用 qwen-plus 生成候选诊断的鉴别诊断 snippet（其余逻辑同父类）。"""
        try:
            resp = _llm_chat(
                self._ddx_gen_client,
                self._ddx_gen_model,
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

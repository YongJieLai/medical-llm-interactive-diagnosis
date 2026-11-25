"""P2 不泄题版：PrimeKG / HPO 表型清单 grounding，替代 oracle DDX-KB。

`RefineCVQwenGapRefutePrimeKGScenario` 与 `RefineCVQwenGapRefuteRAGScenario`
的唯一区别是 grounding 知识源：
  - oracle 版  : differential_diagnoses + distinguishing_feature（答案病子集，泄题）
  - 本版       : disease -> phenotypes（HPO 官方 annotation，独立于 MedQA 答案）

注意语义差异：HPO 给的是「该病的典型表型」，不是「相似病清单 + 区分点」。
因此注入文案从「从清单里挑 alternative」改为「用表型校验矛盾 + 提出反证」，
这正是 P2 文档 §五 第 4 条主张的 counter-evidence 优先设计。
"""
from __future__ import annotations

import os

from eid.scenarios.refine_cv_ablations import RefineCVQwenGapRefuteScenario

try:
    from primekg_ground import PhenotypeGrounder
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from primekg_ground import PhenotypeGrounder


class RefineCVQwenGapRefutePrimeKGScenario(RefineCVQwenGapRefuteScenario):
    """GAP + REFUTATION + PrimeKG/HPO 表型 grounding（不泄题，P2 正式版）。"""

    ENABLE_RAG = True
    HPOA_PATH = os.environ.get("HPOA_PATH", "")
    HP_JSON_PATH = os.environ.get("HP_JSON_PATH", "")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grounder: PhenotypeGrounder | None = None

    def _get_grounder(self) -> PhenotypeGrounder:
        if self._grounder is None:
            self._grounder = PhenotypeGrounder(self.HPOA_PATH, self.HP_JSON_PATH)
        return self._grounder

    def _retrieve_ddx(self, diagnosis: str) -> str:
        """Grounding 检索：返回候选诊断的 HPO 典型表型清单（无则空串）。"""
        return self._get_grounder().retrieve(diagnosis)

    def _rag_inject(self, snippet: str) -> str:
        """表型清单的注入语义：校验矛盾 + 提反证，而非「从清单挑 alternative」。"""
        return (
            "\n\n### External Phenotype Knowledge (authoritative, HPO) ###\n"
            "The candidate diagnosis typically presents with the phenotypes below. "
            "Use them to check whether the evidence gathered so far CONTRADICTS the "
            "candidate diagnosis. If a contradiction exists, state the counter-evidence "
            "and propose a clinically valid alternative diagnosis.\n"
            f"{snippet}"
        )

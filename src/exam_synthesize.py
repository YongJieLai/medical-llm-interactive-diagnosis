"""SDBench 式「诊断中性合成结果」生成器（P1 后续：替代 Not specified 泄漏通道）。

背景
----
`exam_facts` 是"答案导向"构建的（确诊需要什么检查就放什么），因此 closed-world 下
reporter 对"查不到"的检查返回 'Not specified'，这个信号本身 ≈ "该检查与本题答案无关"，
构成泄漏通道（见 docs/检查可得性泄漏_关键发现与业界做法调研.md）。

本模块落地 SDBench（Nori et al., Science 2026）的 Gatekeeper 设计：对查不到的检查，
返回"与病例临床表现一致、但不指向确诊"的合成结果（diagnostically neutral synthetic
findings），而不是 'Not specified'。同时由调用方配 imputation ledger 记账，供 Judge
审计"医生是否靠合成信息答对"。

与 SDBench 的适配点（用户判断）：
  SDBench 因数据检查做得全，可用中性合成；我们的原始数据检查太少，强相关应呈阳性的
  检查若显示中性会误导医生。因此本方案的前提是**完整检查面板**（src/gen_exam_panel.py
  已把常规/排除检查补全，锚定确诊检查不改写）。在面板基础上，"面板外检查"主要是
  非常规/冷门检查，返"诊断中性"的误导风险已被面板大幅压低。

关键约束（防泄露 + 防误导）：
  1. 输入仅 patient_facts（症状）+ exam_facts（已有面板），**绝不输入 ground_truth**。
  2. 诊断中性：不合成病征性（pathognomonic）阳性；不给出能直接确诊的发现。
  3. 防误导：与病例症状相关的检查给"非特异性异常"，只有明确无关才可报正常/阴性。
     （区别于 open-world 方案 A 的"无脑 Normal"——那是系统性撒谎。）
  4. 不标注合成（医生不可见），由 ledger 兜底公平性。

环境变量：
  SYNTHESIZE_MISSING=1  开启合成；缺省关闭（保持 closed-world 口径可回退）。
  SYNTHESIZE_PROVIDER    合成用 LLM 提供方，缺省 deepseek（与 verifier 同模型）。
"""
from __future__ import annotations

import os

_SYS = (
    "You are a clinical findings synthesizer for a diagnostic environment. "
    "Given a test the physician requested that has NO recorded result in the "
    "case, you produce a single diagnostically NEUTRAL synthetic finding that "
    "is consistent with the patient's known presentation."
)

_USER = """Requested test: {test}

Patient's known clinical presentation (symptoms / signs):
{patient_facts}

The patient's already-recorded findings:
{exam_facts}

Produce ONE concise finding for the requested test. Strict rules:
1. Be consistent with the presentation above (do not contradict it).
2. DEFAULT to a non-specific, indeterminate, or borderline result phrased to
   NEITHER support NOR exclude any diagnosis (e.g. "mild, non-specific; correlate
   clinically"). This is your default for ANY test that could plausibly relate to
   the patient's chief complaint or abnormalities.
3. Report a normal/negative value ONLY for routine tests that are CLEARLY
   unrelated to the patient's chief complaint and abnormalities. If in doubt
   whether the test is related, you MUST use the non-specific default.
4. NEVER state a pathognomonic / classic / disease-defining finding, and NEVER
   report a definitive "normal"/"negative" for a test that could exclude a
   diagnosis related to this patient's presentation.
5. Output ONLY the finding itself (a single sentence or a short value+qualifier).
   No explanation, no hedging meta-text.
"""


def _llm(client, model, messages) -> str:
    try:
        from llm_client import chat as _chat
    except ImportError:  # pragma: no cover
        import sys
        sys.path.insert(0, "/workspace/medical_llm_exp/src")
        from llm_client import chat as _chat
    try:
        resp = _chat(client, model, messages, temperature=0.0, max_tokens=256)
    except Exception:  # fail-open：合成失败退回原样
        return ""
    if isinstance(resp, dict):
        return ""
    return (resp or "").strip()


class Synthesizer:
    """懒加载的合成器；仅当环境变量 SYNTHESIZE_MISSING 开启时才真正调用 LLM。"""

    def __init__(self) -> None:
        self._enabled = os.environ.get("SYNTHESIZE_MISSING", "") == "1"
        self._client = None
        self._model = None
        self._provider = os.environ.get("SYNTHESIZE_PROVIDER", "deepseek")

    @classmethod
    def from_env(cls) -> "Synthesizer":
        return cls()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from llm_client import get_llm
        except ImportError:  # pragma: no cover
            import sys
            sys.path.insert(0, "/workspace/medical_llm_exp/src")
            from llm_client import get_llm
        try:
            self._client, self._model = get_llm(self._provider)
            return True
        except Exception:  # pragma: no cover
            self._client = False
            return False

    def synthesize(self, test: str, patient_facts: str, exam_facts: str) -> str:
        """返回合成结果；未开启 / 失败时返回空串（调用方回退 Not specified）。"""
        if not self._enabled:
            return ""
        if not test:
            return ""
        if not self._ensure_client():
            return ""
        user = _USER.format(
            test=test,
            patient_facts=patient_facts or "(none recorded)",
            exam_facts=exam_facts or "(none recorded)",
        )
        return _llm(self._client, self._model, [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": user},
        ])

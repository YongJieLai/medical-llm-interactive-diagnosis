"""规则脚本患者模拟器。

用 DDXPlus 的结构化证据表充当"患者真相"：
- 开场只主动说 INITIAL_EVIDENCE 对应的主诉
- 医生指定问某个证据（evidence_name），患者精确查表回答（阳性给值 / 阴性说 No）
- 不编造、不漏说，保证患者侧信息释放是精确无偏的
"""
from .ddxplus_loader import evidence_answer, negative_answer, parse_evidence_token


class ScriptPatient:
    def __init__(self, case, evidences):
        self.case = case
        self.ev = evidences
        self.age = case["age"]
        self.sex = case["sex"]
        self.evidence_values = case["evidence_values"]  # 隐藏真相 {E_xx: True/'V_xx'/'6'}
        self.asked = set()

    def demographics(self) -> str:
        sex = "male" if self.sex == "M" else "female"
        return f"Age {self.age}, {sex}."

    def initial_utterance(self) -> str:
        return " ".join(self.case["initial_facts"])

    def answer(self, evidence_name: str) -> str:
        self.asked.add(evidence_name)
        if evidence_name in self.evidence_values:
            return evidence_answer(self.ev, evidence_name, self.evidence_values[evidence_name])
        return negative_answer(self.ev, evidence_name)

    def answer_batch(self, evidence_names):
        return [(n, self.answer(n)) for n in evidence_names]

    def covered_evidence(self):
        """已获取到的阳性证据（用于算信息覆盖率）。"""
        return set(self.evidence_values.keys()) & self.asked

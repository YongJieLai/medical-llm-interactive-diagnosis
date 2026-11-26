"""MEDIQ abstention scenario — Scale + Rationale Generation + Self-Consistency（零训练移植）。

忠实移植 NeurIPS 2024 "MEDIQ: Question-Asking LLMs and a Benchmark for Reliable
Interactive Clinical Reasoning"（arXiv:2406.00922，开源 stellalisy/mediQ）的 Expert
系统 abstention 策略：论文最强组合 = Scale(1-5 Likert) + Rationale Generation + Self-Consistency。

核心循环（每轮）：
  1. Scale + RG：LLM 评估置信度（1-5 Likert）+ rationale（说明缺什么信息）
  2. Self-Consistency：多次采样取平均 Likert 分数
  3. 平均分 >= 4.0 → 给最终诊断（FINISH）
  4. 平均分 <  4.0 → 生成一个 atomic question，路由 patient/reporter 提问

场景适配（唯一改动）：
  - 原文是 MCQ（选项 A/B/C/D + letter choice）；开放式诊断无选项，把「pick the correct
    option」改为「make the correct diagnosis」，最终输出改为自由文本诊断名。
  - 提问路由 patient/reporter（见 _ext_baselines_common）。
"""

from __future__ import annotations

import re

from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

from llm_client import get_llm
from eid.scenarios._ext_baselines_common import (
    build_simulators, ask_question, elicit_chief_complaint)

SCALE_THRESHOLD = 4.0
N_SELF_CONSISTENCY = 3

LIKERT_MAP = {
    "very confident": 5,
    "somewhat confident": 4,
    "neither confident or unconfident": 3,
    "somewhat unconfident": 2,
    "very unconfident": 1,
}

# 原文 scale_RG prompt，MCQ → 开放式诊断改写
SCALE_RG_PROMPT = """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to make the correct diagnosis factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct diagnosis is supported by all evidence, and there is enough evidence to eliminate the rest of the candidates, so the diagnosis can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct diagnosis is more likely than others, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There is evidence supporting the correct diagnosis, but further evidence is needed to be sure which one is the correct diagnosis.
"Somewhat Unconfident" - There is evidence supporting more than one diagnosis, therefore more questions are needed to further distinguish the candidates.
"Very Unconfident" - There is not enough evidence supporting any diagnosis, the likelihood of making the correct diagnosis at this point is near random guessing.

Answer in the following format:
REASON: a one-sentence explanation of why you are or are not confident and what other information is needed.
DECISION: chosen rating from the above list."""

# 原文 atomic_question_improved prompt，MCQ → 开放式诊断改写
ATOMIC_QUESTION_PROMPT = """If there are missing features that prevent you from making a confident and factual diagnosis, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. You can ask about any relevant information about the patient's case, such as family history, tests and exams results, treatments already done, etc. Consider what are the common questions asked in the specific subject relating to the patient's known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:
ATOMIC QUESTION: the atomic question and NOTHING ELSE."""

# 原文 answer prompt，MCQ → 开放式诊断改写
ANSWER_PROMPT = """Assume that you already have enough information from the above question-answer pairs to make a diagnosis, use the above information to produce a factual conclusion. Respond with the diagnosis name and NOTHING ELSE."""

CURR_TEMPLATE = """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {patient_info}
CONVERSATION LOG:
{conv_log}
YOUR TASK: {task}"""


def parse_likert(text):
    t = text.lower()
    for k, v in LIKERT_MAP.items():
        if k in t:
            return v
    return 0


class MediQAbstentionScenario(BaseScenario):
    """MEDIQ abstention：Scale + RG + Self-Consistency 决定「何时问 / 何时答」。"""

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config,
                 max_turns=16, min_turns=0):
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.min_turns = min_turns  # 强制至少问 min_turns 轮（对照：观察提问质量）
        self.prompts = PromptManager(dataset_name)
        self.client, self.model = get_llm("qwen")

    def _llm(self, messages, temperature=0.0):
        from llm_client import chat
        return chat(self.client, self.model, messages,
                    temperature=temperature, max_tokens=1024)

    def _scale_rg(self, patient_info, conv_log):
        task = SCALE_RG_PROMPT
        user = CURR_TEMPLATE.format(patient_info=patient_info,
                                    conv_log=conv_log or "None", task=task)
        return self._llm([{"role": "user", "content": user}])

    def _self_consistency_score(self, patient_info, conv_log):
        scores = []
        reasons = []
        for _ in range(N_SELF_CONSISTENCY):
            r = self._scale_rg(patient_info, conv_log)
            scores.append(parse_likert(r))
            m = re.search(r"REASON:\s*(.+)", r, re.DOTALL)
            if m:
                reasons.append(m.group(1).strip())
        avg = sum(scores) / len(scores) if scores else 0
        return avg, reasons, scores

    def _gen_question(self, patient_info, conv_log):
        user = CURR_TEMPLATE.format(patient_info=patient_info,
                                    conv_log=conv_log or "None", task=ATOMIC_QUESTION_PROMPT)
        r = self._llm([{"role": "user", "content": user}])
        m = re.search(r"ATOMIC QUESTION:\s*(.+)", r, re.DOTALL)
        return (m.group(1).strip() if m else r.strip())

    def _gen_answer(self, patient_info, conv_log):
        user = CURR_TEMPLATE.format(patient_info=patient_info,
                                    conv_log=conv_log or "None", task=ANSWER_PROMPT)
        r = self._llm([{"role": "user", "content": user}])
        return r.strip()

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient, reporter = build_simulators(
            case_input, self.patient_config, self.reporter_config, self.prompts)

        trace = []
        # 空起步（对齐上游交互场景）：第一轮问主诉，主诉作为患者自述起点。
        # 病史细节与检查结果只能通过后续提问获取（不再用全信息 task 开卷）。
        chief = elicit_chief_complaint(patient, self.prompts, trace)
        patient_info = chief
        conv_log = ""
        asked = []
        answer = ""
        current_turn = 1

        while current_turn < self.max_turns:
            # 1. Scale + RG + Self-Consistency 评估置信度
            conf, reasons, scores = self._self_consistency_score(patient_info, conv_log)
            trace.append({
                "role_id": "doctor_abstain",
                "content": f"confidence={conf:.2f} scores={scores} reasons={reasons}",
            })

            # 2. 置信度达标 → 给诊断（min_turns 内强制继续问）
            if conf >= SCALE_THRESHOLD and current_turn >= self.min_turns:
                answer = self._gen_answer(patient_info, conv_log)
                trace.append({"role_id": "doctor", "content": answer, "duration": 0.0})
                break

            # 3. 置信度不足 → 生成 atomic question，问 patient/reporter
            q = self._gen_question(patient_info, conv_log)
            if not q or q in asked:
                answer = self._gen_answer(patient_info, conv_log)
                trace.append({"role_id": "doctor", "content": answer, "duration": 0.0})
                break

            trace.append({"role_id": "doctor", "content": q, "duration": 0.0})
            asked.append(q)
            conv_log += f"\nDoctor: {q}\n"

            resp, role = ask_question(q, patient, reporter, self.prompts, trace)
            if role == "reporter":
                conv_log += f"\nMeasurement: {resp}\n"
            else:
                conv_log += f"\nPatient: {resp}\n"

            current_turn += 1

        # 到轮数上限仍未达标 → 强制给诊断
        if not answer:
            answer = self._gen_answer(patient_info, conv_log)
            trace.append({"role_id": "doctor", "content": answer, "duration": 0.0})

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "mediq_abstention",
                "turns": current_turn,
                "max_turns": self.max_turns,
            },
        )


class MediQAbstentionMinQScenario(MediQAbstentionScenario):
    """MEDIQ abstention 强制交互对照：至少问 4 轮（观察提问质量，绕过 qwen 过早自信停止）。"""

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config,
                 max_turns=16):
        super().__init__(dataset_name, doctor_config, patient_config, reporter_config,
                         max_turns=max_turns, min_turns=4)

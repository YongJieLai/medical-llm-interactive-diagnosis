"""外部 baseline（MedClarify / MEDIQ-abstention / UoT）共享基础设施。

这三个方法复用 EID 的 patient / reporter simulator（保证患者模拟一致、`[REFERENCE]`
编号可被 coverage 分析复用），只替换 doctor 的决策循环。doctor 决策循环用 llm_client
直接调 qwen（方法特定的多步 LLM 调用）。

场景适配（唯一改动点，其余忠实还原各方法）：
  - 三个方法原文都只有一个"回答者"（MCQ 的 patient record），而 EID 场景把信息拆成
    patient_facts（病史）+ exam_facts（检查）两个源。这里用「关键词预判 + patient N/A
    兜底」把自由文本问题路由到 patient 或 reporter。
"""

from __future__ import annotations

import re

from eid.agents import create_agent

# 开场主诉问题（对齐上游：doctor 空起步，第一轮主动问患者主诉）。
# 上游 roleplay/react/sc/refine 的 doctor 第一轮 last_reply 为空、无任何病例信息，
# 必须自己 [QUERY] 问主诉。外部方法移植也必须空起步，不能用 case_input.task（全信息开卷）。
CHIEF_COMPLAINT_Q = (
    "What brings you in today? Please describe your main symptoms and complaints."
)


def elicit_chief_complaint(patient, prompts, trace) -> str:
    """第一轮问主诉，返回患者 [RESPONSE] 主诉文本，作为后续方法的 patient_info/repo 起点。

    这等价于上游交互场景 doctor 的第 1 个 turn（问主诉），使外部方法与我们的
    baseline（roleplay/react/sc/refine）在「起步信息量」上完全对齐：双方都只能
    通过提问获取病史与检查证据。
    """
    trace.append({"role_id": "doctor", "content": CHIEF_COMPLAINT_Q, "duration": 0.0})
    instr = prompts.get_patient_turn_instruction(CHIEF_COMPLAINT_Q)
    resp = patient.step(instr)
    trace.append({"role_id": "patient", "content": resp,
                  "duration": patient.get_last_duration()})
    m = re.search(r"\[RESPONSE\]\s*([\s\S]*)", resp)
    return m.group(1).strip() if m else resp

# 检查类问题关键词（命中 → 路由到 reporter 开检查；否则 → patient 问病史）
EXAM_KEYWORDS = [
    "lab", "laboratory", "test", "testing", "imaging", " ct", "mri", "x-ray", "x ray",
    "ultrasound", "ultrasonography", "echocardiogram", "ecg", "ekg", "electrocardiogram",
    "culture", "biopsy", "histology", "histopathology", "antibody", "serology", "antigen",
    "blood", "serum", "urine", "urinalysis", "cbc", "count", "level", "titer", "pcr",
    "endoscopy", "colonoscopy", "bronchoscopy", "lumbar puncture", "spinal tap",
    "radiograph", "stool", "sputum", "gram stain", "smear", "pathology", "assay",
    "angiogram", "angiography", "scan", "film", "检查", "化验", "影像", "培养", "活检",
    "抗体", "血", "尿", "心电图", "超声",
]


def is_exam_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in EXAM_KEYWORDS)


def extract_patient_reference(text: str) -> str:
    """从 patient 回复里提取 [REFERENCE] 部分（判断是否 N/A）。"""
    m = re.search(r"\[REFERENCE\]\s*(.*?)(?=\n\[RESPONSE\]|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_simulators(case_input, patient_config, reporter_config, prompts):
    """创建 EID patient / reporter agent（与 refine_clean 完全一致）。"""
    patient = create_agent(
        role_id="patient",
        system_prompt=prompts.get_patient_system_prompt().format(
            patient_facts="\n".join(case_input.patient_facts)
        ),
        config=patient_config,
        message_window_size=1,
        summarize_threshold=95,
    )
    reporter = create_agent(
        role_id="reporter",
        system_prompt=prompts.get_reporter_system_prompt().format(
            exam_facts="\n".join(case_input.exam_facts)
        ),
        config=reporter_config,
        message_window_size=1,
        summarize_threshold=80,
    )
    return patient, reporter


def ask_question(question, patient, reporter, prompts, trace):
    """把一个问题路由到 patient 或 reporter，返回 (回答全文, role_id)。

    - 检查类问题（关键词）→ reporter
    - 其余 → patient；若 patient 返回 [REFERENCE] N/A（说明其实是检查类），兜底转 reporter
    """
    if is_exam_question(question):
        instr = prompts.get_reporter_turn_instruction(question)
        resp = reporter.step(instr)
        trace.append({
            "role_id": "reporter",
            "content": resp,
            "duration": reporter.get_last_duration(),
        })
        return resp, "reporter"

    instr = prompts.get_patient_turn_instruction(question)
    resp = patient.step(instr)
    trace.append({
        "role_id": "patient",
        "content": resp,
        "duration": patient.get_last_duration(),
    })

    ref = extract_patient_reference(resp)
    if ref.upper() == "N/A":
        instr2 = prompts.get_reporter_turn_instruction(question)
        resp2 = reporter.step(instr2)
        trace.append({
            "role_id": "reporter",
            "content": resp2,
            "duration": reporter.get_last_duration(),
        })
        return resp2, "reporter"
    return resp, "patient"

"""AgentClinic 环境适配：LLM 患者 + 测量 agent + 自由问诊医生（vanilla）。

复用 AgentClinic 原版的病例数据与三 agent prompt 设计，但：
- 用我们自己的 llm_client（国产 OpenAI 兼容 API，可配 base_url）
- 采集结构化 trace（每轮的医生动作、患者/测量回答），便于失效分析
- 患者/测量/医生用同一个 provider（先千问），后续可拆分

病例结构（agentclinic_medqa.jsonl 每条）：
  OSCE_Examination = {
    Objective_for_Doctor: 主诉/医生任务,
    Patient_Actor: {Demographics, History, Symptoms, Past_Medical_History, ...},
    Physical_Examination_Findings: {Vital_Signs, ...},
    Test_Results: {Blood_Tests, Imaging, ...},
    Correct_Diagnosis: 正确答案,
  }
"""
import json
import re
from pathlib import Path

from .llm_client import chat

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "agentclinic"


# ---------- 病例加载 ----------

def load_medqa_cases(split="medqa", limit=None):
    """split: medqa(107) / medqa_extended(215) / nejm(15) / nejm_extended(120)。"""
    path = BENCH_DIR / f"agentclinic_{split}.jsonl"
    cases = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        d = json.loads(line)["OSCE_Examination"]
        cases.append({
            "case_id": f"{split}_{i}",
            "objective": d["Objective_for_Doctor"],
            "patient_info": d["Patient_Actor"],
            "physical_exams": d.get("Physical_Examination_Findings", {}),
            "tests": d.get("Test_Results", {}),
            "diagnosis": d["Correct_Diagnosis"],
        })
        if limit and len(cases) >= limit:
            break
    return cases


# ---------- Prompt（复刻 AgentClinic 原版设计） ----------

PATIENT_SYSTEM = (
    "You are a patient in a clinic who only responds in the form of dialogue. "
    "You are being inspected by a doctor who will ask you questions and will perform exams on you "
    "in order to understand your disease. Your answer will only be 1-3 sentences in length.\n\n"
    "Below is all of your information. {info}\n\n"
    "Remember, you must not reveal your disease explicitly but may only convey the symptoms you have "
    "in the form of dialogue if you are asked."
)

MEASURE_SYSTEM = (
    "You are a measurement reader who responds with medical test results. "
    'Please respond in the format "RESULTS: [results here]"\n\n'
    "Below is all of the information you have. {info}\n\n"
    "If the requested results are not in your data then you can respond with NORMAL READINGS."
)


def doctor_system(objective, max_turns, asked_so_far, img_request=False):
    img = ("You may also request medical images related to the disease to be returned with "
           '"REQUEST IMAGES".' if img_request else "")
    return (
        "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
        "You are inspecting a patient who you will ask questions in order to understand their disease. "
        f"You are only allowed to ask {max_turns} questions total before you must make a decision. "
        f"You have asked {asked_so_far} questions so far. "
        'You can request test results using the format "REQUEST TEST: [test]". '
        'For example, "REQUEST TEST: Chest_X-Ray". '
        "Your dialogue will only be 1-3 sentences in length. "
        'Once you have decided to make a diagnosis please type "DIAGNOSIS READY: [diagnosis here]".'
        + img +
        f"\n\nBelow is all of the information you have. {objective}\n\n"
        "Remember, you must discover their disease by asking them questions. "
        "You are also able to provide exams."
    )


def _fmt(d):
    return json.dumps(d, ensure_ascii=False, indent=2)


# ---------- 患者 agent ----------

class LLMPatient:
    """LLM 扮演患者：持完整患者信息，按医生提问以自然语言回答，不主动透露病名。"""

    def __init__(self, client, model, case):
        self.client = client
        self.model = model
        self.info = case["patient_info"]
        self.hist = []  # [(doctor_utterance, patient_answer)]

    def answer(self, doctor_utterance):
        msgs = [{"role": "system",
                 "content": PATIENT_SYSTEM.format(info=_fmt(self.info))}]
        for dq, pa in self.hist:
            msgs.append({"role": "user", "content": dq})
            msgs.append({"role": "assistant", "content": pa})
        msgs.append({"role": "user", "content": doctor_utterance})
        ans = chat(self.client, self.model, msgs, temperature=0.3, max_tokens=150)
        self.hist.append((doctor_utterance, ans))
        return ans


# ---------- 测量 agent ----------

class MeasurementAgent:
    """LLM 测量 agent：持体格检查 + 化验/影像结果，医生开检查时返回对应结果。"""

    def __init__(self, client, model, case):
        self.client = client
        self.model = model
        self.info = {**case["physical_exams"], "tests": case["tests"]}
        self.hist = []

    def read(self, test_request):
        msgs = [{"role": "system",
                 "content": MEASURE_SYSTEM.format(info=_fmt(self.info))}]
        for q, a in self.hist:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": test_request})
        ans = chat(self.client, self.model, msgs, temperature=0.0, max_tokens=150)
        self.hist.append((test_request, ans))
        return ans


# ---------- 医生循环（vanilla 自由问诊） ----------

RE_TEST = re.compile(r"REQUEST TEST\s*:\s*(.+)", re.IGNORECASE)
RE_DIAG = re.compile(r"DIAGNOSIS READY\s*:\s*(.+)", re.IGNORECASE)


def run_vanilla(client, model, case, max_turns=20, verbose=False):
    """自由问诊医生：每轮生成"问问题 / 开检查 / 下诊断"，与患者、测量交互，采集 trace。"""
    patient = LLMPatient(client, model, case)
    measurement = MeasurementAgent(client, model, case)
    objective = case["objective"]

    trace = []
    hist = []  # 对话历史（给医生的上下文）
    diagnosis = None

    for turn in range(max_turns):
        sysp = doctor_system(objective, max_turns, turn)
        msgs = [{"role": "system", "content": sysp}]
        for q, a in hist:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user",
                     "content": "(Please continue your dialogue as the doctor.)"})

        out = chat(client, model, msgs, temperature=0.2, max_tokens=200)

        m_test = RE_TEST.search(out)
        m_diag = RE_DIAG.search(out)

        if m_diag:
            diagnosis = m_diag.group(1).strip()
            trace.append({"turn": turn, "action": "diagnose",
                          "doctor": out, "diagnosis": diagnosis})
            break
        elif m_test:
            test_name = m_test.group(1).strip()
            result = measurement.read(test_name)
            trace.append({"turn": turn, "action": "request_test",
                          "doctor": out, "test": test_name, "result": result})
            hist.append((out, result))
        else:
            ans = patient.answer(out)
            trace.append({"turn": turn, "action": "ask",
                          "doctor": out, "patient": ans})
            hist.append((out, ans))

        if verbose:
            a = trace[-1]["action"]
            print(f"  [t{turn}] {a}: {out[:80]}")

    if diagnosis is None:
        # 到上限未下诊断：强制让医生给一个诊断
        msgs = [{"role": "system", "content": doctor_system(objective, max_turns, max_turns)}]
        for q, a in hist:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user",
                     "content": "You must now give a diagnosis. Reply with 'DIAGNOSIS READY: [diagnosis]'."})
        out = chat(client, model, msgs, temperature=0.0, max_tokens=200)
        m_diag = RE_DIAG.search(out)
        diagnosis = m_diag.group(1).strip() if m_diag else out.strip()
        trace.append({"turn": max_turns, "action": "diagnose_forced", "doctor": out,
                      "diagnosis": diagnosis})

    return {
        "method": "vanilla",
        "case_id": case["case_id"],
        "diagnosis": diagnosis,
        "truth": case["diagnosis"],
        "turns": len(trace),
        "trace": trace,
    }


# ---------- 裁判（先规则匹配，后续可换 LLM 判定） ----------

def normalize(s):
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def is_correct(diag, truth):
    if not diag:
        return False
    d, t = normalize(diag), normalize(truth)
    return d == t or d in t or t in d

"""UoT scenario — 不确定性感知模拟 + 信息增益选问题（零训练移植）。

忠实移植 arXiv:2402.03271 "Uncertainty of Thoughts: Uncertainty-Aware Planning
Enhances Information Seeking in Large Language Models"（开源 zhiyuanhubj/UoT）的
medical diagnosis 任务（open-set 模式）。

核心循环（每轮）：
  1. 维护「可能性空间」= 候选疾病集合（open-set：LLM 生成 size 个候选）
  2. 生成 n 个候选 YES/NO 问题，每个问题把候选集分成 YES/NO 两组（尽量平分）
  3. 信息增益奖励 reward_function(x)= H(x)/(1+|2x-1|/λ)，x=YES 侧占比（平分时最大）
  4. 选 idiv_reward 最大的问题 → 问 patient
  5. 用 classify 把「问题+回答」映射回候选集，YES 组作为新候选集
  6. 候选集过小时 renew_open_set 重新生成
  7. 停止：|items|<=2 或 max_turn；最终诊断 = 剩余候选中 LLM 选最可能

场景适配（唯一改动）：
  - 原文 examiner 是「患有目标病的模拟患者」、答 Yes/No；这里用 EID patient（自由文本），
    用 free_answer 模式的 classify 逻辑（cls_given_repo）把回答映射回候选集。
  - 提问路由 patient/reporter（见 _ext_baselines_common）。
  - n_extend_layers 取 1（原文多步树扩展在开放式 200 例下计算量爆炸，1 层已保留
    「信息增益选问题」核心，见文档标注）。
"""

from __future__ import annotations

import math
import re

from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

from llm_client import get_llm
from eid.scenarios._ext_baselines_common import (
    build_simulators, ask_question, elicit_chief_complaint)

OPEN_SET_SIZE = 8
N_POTENTIAL_ACTIONS = 3
LAMB = 0.4

# 原文 init_open_set_prompt
INIT_OPEN_SET_PROMPT = """You are a doctor and your patient self-reports that: {repo}. Please propose {size} diseases that you think your patient may suffer from.
Your response should be: ["disease1", "disease2", ...]"""

# 原文 renew_open_set_prompt
RENEW_OPEN_SET_PROMPT = """Based on the conversation history, please propose {size} diseases that your patient may suffer from.
The list of {size} diseases should contains {item_list}
Your response should be: ["disease1", "disease2", ...]"""

# 原文 generate_prompt（JSON 输出约束——qwen-plus 为推理模型会输出长 CoT，
# 原文「Question i / YES / NO」自由文本格式会被 CoT 打乱，故改用 JSON 约束输出，
# 语义与原文一致：设计 n 个 YES/NO 问题、把候选病尽量平分成 YES/NO 两组）
GENERATE_PROMPT = """You are a doctor. Here are all the possible diseases that the patient may suffer from:
{items_str}

Design {n} YES/NO questions to ask the patient with symptoms. Each question must be answerable by YES or NO only, and must split the diseases above into a YES group and a NO group as evenly as possible (the count of YES and NO should differ by no more than one). {asked}

For each question, classify every disease above: if the answer to the question would be YES for that disease, put it into "items_yes"; otherwise put it into "items_no".

Output JSON only, with key "questions": a list of {n} objects, each with keys:
- "question": str (the YES/NO question)
- "items_yes": list of disease names
- "items_no": list of disease names"""

# 原文 classify_prompt（free_answer 模式 cls_given_repo）
CLASSIFY_PROMPT = """Here are all diseases that the patient may suffer from:
{item_list_str}

{repo}
For each disease under this report, if the patient is possible to have, put this disease into 'YES: ...', otherwise to 'NO: ...'. And your answer should be like:
YES: aaa, bbb, ... (disease names only)
NO: ccc, ddd, ... (disease names only)"""


def reward_function(x, lamb=LAMB):
    if x in (0, 1):
        return 0.0
    h = -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    return h / (1 + abs(2 * x - 1) / lamb)


class UoTScenario(BaseScenario):
    """UoT：不确定性感知模拟 + 信息增益选问题（open-set 医学诊断）。"""

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config,
                 max_turns=16):
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.prompts = PromptManager(dataset_name)
        self.client, self.model = get_llm("qwen")

    def _llm(self, messages, temperature=0.0, max_tokens=2048):
        from llm_client import chat
        return chat(self.client, self.model, messages,
                    temperature=temperature, max_tokens=max_tokens)

    def _init_open_set(self, repo):
        msg = [{"role": "user", "content": INIT_OPEN_SET_PROMPT.format(
            repo=repo or "None", size=OPEN_SET_SIZE)}]
        r = self._llm(msg)
        return self._parse_list(r)

    def _renew_open_set(self, conv_log, items):
        msg = [{"role": "user", "content": RENEW_OPEN_SET_PROMPT.format(
            size=OPEN_SET_SIZE, item_list=str(items))}]
        r = self._llm(msg)
        new_items = self._parse_list(r)
        # 保留已有候选 + 新候选
        merged = list(dict.fromkeys(items + new_items))
        return merged[:OPEN_SET_SIZE] if merged else items

    def _parse_list(self, text):
        try:
            lst = eval(text.strip())
            if isinstance(lst, list):
                return [str(x) for x in lst if str(x).strip()]
        except Exception:
            pass
        # 兜底：提取引号内字符串
        return re.findall(r"[\"']([^\"']+)[\"']", text)

    def _generate_questions(self, items, asked):
        asked_str = ("(The question should not be '" + "' or '".join(asked) + "')"
                     if asked else "")
        msg = [{"role": "user", "content": GENERATE_PROMPT.format(
            items_str=", ".join(items), n=N_POTENTIAL_ACTIONS, asked=asked_str)}]
        from llm_client import chat
        r = chat(self.client, self.model, msg, temperature=0.0, max_tokens=2000, json_mode=True)
        if isinstance(r, dict) and "questions" in r:
            return [{"question": c.get("question", "").strip(),
                     "items_yes": list(c.get("items_yes", [])),
                     "items_no": list(c.get("items_no", []))}
                    for c in r["questions"] if c.get("question")]
        # 兜底：文本解析
        return self._process_questions(r if isinstance(r, str) else str(r))

    def _process_questions(self, rsp):
        """兜底解析 UoT generate 文本输出：Question i / YES / NO。"""
        ans = []
        for i in range(N_POTENTIAL_ACTIONS):
            marker = f"Question {i + 1}: "
            if marker not in rsp:
                continue
            seg = rsp.split(marker, 1)[1]
            q = seg.split("\n", 1)[0].strip()
            if "YES: " not in seg:
                continue
            yes_seg = seg.split("YES: ", 1)[1]
            yes_line = yes_seg.split("\n", 1)[0]
            items_yes = [x.strip() for x in yes_line.split(", ") if x.strip()]
            no_seg = seg.split("NO: ", 1)[1] if "NO: " in seg else seg.split("\nNO: ", 1)[1]
            no_line = no_seg.split("\n", 1)[0]
            items_no = [x.strip() for x in no_line.split(", ") if x.strip()]
            ans.append({"question": q, "items_yes": items_yes, "items_no": items_no})
        return ans

    def _classify(self, items, repo):
        msg = [{"role": "user", "content": CLASSIFY_PROMPT.format(
            item_list_str=", ".join(items), repo=repo)}]
        r = self._llm(msg, max_tokens=len(items) * 20 + 200)
        yes, no = [], []
        if "YES: " in r:
            yes_seg = r.split("YES: ", 1)[1].split("\n", 1)[0]
            yes = [x.strip() for x in yes_seg.split(", ") if x.strip()]
        if "NO: " in r:
            no_seg = r.split("NO: ", 1)[1].split("\n", 1)[0]
            no = [x.strip() for x in no_seg.split(", ") if x.strip()]
        return yes, no

    def _final_diagnosis(self, items, conv_log):
        msg = [{"role": "user", "content": (
            "You are a doctor. The patient may suffer from one of these diseases: "
            f"{', '.join(items) or 'an unknown disease'}.\n\nConversation:\n{conv_log}\n\n"
            "Based on the conversation, give the single most likely diagnosis. "
            "Respond with the disease name and NOTHING ELSE."
        )}]
        return self._llm(msg, max_tokens=200).strip()

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient, reporter = build_simulators(
            case_input, self.patient_config, self.reporter_config, self.prompts)

        trace = []
        # 空起步（对齐上游交互场景）：第一轮问主诉，用主诉做 open-set 初始化，
        # 而非全信息 task。病史细节与检查结果只能通过后续提问获取。
        repo = elicit_chief_complaint(patient, self.prompts, trace)
        conv_log = f"Patient: {repo}\n"
        asked = []
        answer = ""

        # 1. open-set 初始化（以主诉为起点）
        items = self._init_open_set(repo)
        if not items:
            items = self._init_open_set(repo)  # 重试一次
        trace.append({"role_id": "doctor_open_set", "content": str(items)})

        current_turn = 1

        while current_turn < self.max_turns and len(items) > 2:
            # 2. 生成候选问题 + 信息增益评分
            cands = self._generate_questions(items, asked)
            if not cands:
                break

            best = None
            best_reward = -1e9
            for c in cands:
                q = c["question"]
                n_yes = len(c["items_yes"])
                n_no = len(c["items_no"])
                total = n_yes + n_no
                if total == 0:
                    continue
                x = n_yes / total
                rw = reward_function(x)
                trace.append({"role_id": "doctor_uot", "content": (
                    f"q={q} yes={n_yes} no={n_no} idiv_reward={rw:.4f}")})
                if rw > best_reward and q not in asked:
                    best_reward = rw
                    best = c

            if best is None:
                break

            q = best["question"]
            trace.append({"role_id": "doctor", "content": q, "duration": 0.0})
            asked.append(q)
            conv_log += f"\nDoctor: {q}\n"

            # 3. 问 patient/reporter
            resp, role = ask_question(q, patient, reporter, self.prompts, trace)
            if role == "reporter":
                conv_log += f"\nMeasurement: {resp}\n"
            else:
                conv_log += f"\nPatient: {resp}\n"

            # 4. 用 classify 更新候选集（free_answer 逻辑）
            repo = f"Doctor: {q}\nPatient: {resp}\n"
            yes, no = self._classify(items, repo)
            new_items = yes if yes else items
            trace.append({"role_id": "doctor_update", "content": (
                f"yes={yes} no={no} items={len(items)}->{len(new_items)}")})

            # 候选集没缩小 → renew
            if len(new_items) >= len(items):
                new_items = self._renew_open_set(conv_log, items)

            items = new_items
            current_turn += 1

        # 5. 最终诊断
        answer = self._final_diagnosis(items, conv_log)
        trace.append({"role_id": "doctor", "content": answer, "duration": 0.0})

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "uot",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "final_items": items,
            },
        )

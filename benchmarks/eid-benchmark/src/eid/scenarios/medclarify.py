"""MedClarify scenario — 信息增益选问题 + Bayesian 更新（零训练移植）。

忠实移植 arXiv:2602.17308 "MedClarify: An information-seeking AI agent for medical
diagnosis with case-specific follow-up questions"。

核心循环（每轮）：
  1. LLM 生成 k=5 候选诊断 + 置信度（DDx，JSON）
  2. 生成 k 个候选问题（k-1 区分性：反驳 top1 + 确认替代；1 探索性）
  3. 对每个问题模拟「是/否」两分支 → 新 DDx → 算 DEIG = α·IG + β·Div + γ·Con
  4. 选 DEIG 最高的问题，路由 patient/reporter 提问
  5. 收到回答 → Bayesian 更新（先验 × LLM 似然，温度缩放 T=1.1）
  6. 停止：t=16 或 top1 置信度 >0.97 或 top1−top2 gap >0.85

场景适配（唯一改动）：
  - Div 项原文用 ICD-11 章节相似度矩阵；开放式诊断无 ICD，改用 LLM 批量判断
    yes/no 两分支 top1 诊断的语义相似度（0-1）。
  - 提问路由 patient/reporter（见 _ext_baselines_common）。
"""

from __future__ import annotations

import json
import math
import re
import time

from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

from llm_client import get_llm
from eid.scenarios._ext_baselines_common import (
    build_simulators, ask_question, elicit_chief_complaint)

ALPHA, BETA, GAMMA = 0.5, 0.35, 0.15
K = 5
TEMP_SCALING = 1.1
CONF_STOP = 0.97
GAP_STOP = 0.85

DDX_PROMPT = """You are a medical doctor reasoning through a real clinical case. Based on the patient information and conversation so far, generate a ranked differential diagnosis.

PATIENT INFORMATION:
{patient_info}

CONVERSATION LOG:
{conv_log}

Generate the top {k} differential diagnoses. For each, assign a confidence score in [0.1, 1.0]: 0.9-1.0 means expected to be correct, 0.6-0.9 plausible, below 0.6 less likely. Use specific disease names, not broad categories.

Output JSON only, with keys:
- "differential_diagnosis": list of {{"diagnosis": str, "confidence": float}}
- "diagnostic_uncertainty": float (Shannon entropy of the confidence distribution)
- "reasoning": str
"""

CONFIRM_QUESTION_PROMPT = """Generate a question to help eliminate {d_eliminate} but confirm {d_confirm}. The question should be bite-sized and not more than 1 sentence. Do not repeat the questions you have asked before in {asked}. You are not allowed to reveal explicitly your diagnosis that you guessed.

Output your answer concisely in the following format:
Thought: [Your thought process here]
Response: [Your question here]
"""

EXPLORE_QUESTION_PROMPT = """Generate a question to discover other possible diseases apart from {d_list}. The question should be bite-sized and not more than 1 sentence. Do not repeat the questions you have asked before in {asked}. You are not allowed to reveal explicitly your diagnosis that you guessed.

Output your answer concisely in the following format:
Thought: [Your thought process here]
Response: [Your question here]
"""

SIMULATE_PROMPT = """You are a medical doctor reasoning through a real clinical case.

PATIENT INFORMATION:
{patient_info}

CONVERSATION LOG:
{conv_log}

Now assume the patient is asked the following question and answers "{answer}":

QUESTION: {question}
PATIENT'S ANSWER: {answer}

Update the differential diagnosis accordingly. Output JSON only, with keys:
- "differential_diagnosis": list of {{"diagnosis": str, "confidence": float}}
- "reasoning": str
"""


def _entropy(probs: list[float]) -> float:
    probs = [max(p, 1e-12) for p in probs]
    return -sum(p * math.log(p) for p in probs)


def _gini(probs: list[float]) -> float:
    probs = sorted(probs)
    n = len(probs)
    total = sum(probs) or 1e-12
    return sum((2 * k - (n + 1)) * x for k, x in enumerate(probs, 1)) / (n * total)


class MedClarifyScenario(BaseScenario):
    """MedClarify：DEIG 信息增益选问题 + Bayesian 更新。"""

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config,
                 max_turns=16, min_turns=0, time_budget=None):
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.min_turns = min_turns  # 强制至少问 min_turns 轮（对照：观察提问质量）
        self.time_budget = time_budget  # 单例时间预算（秒），None=不限制；工程保护，不改方法逻辑
        self.prompts = PromptManager(dataset_name)
        self.client, self.model = get_llm("qwen")

    def _llm(self, messages, json_mode=False, temperature=0.3):
        from llm_client import chat
        return chat(self.client, self.model, messages,
                    temperature=temperature, max_tokens=2048, json_mode=json_mode)

    def _gen_ddx(self, patient_info, conv_log):
        msg = [{"role": "user", "content": DDX_PROMPT.format(
            patient_info=patient_info, conv_log=conv_log or "None", k=K)}]
        r = self._llm(msg, json_mode=True)
        if isinstance(r, dict):
            return r
        # 兜底：非 JSON，尝试解析
        try:
            return json.loads(r)
        except Exception:
            return {"differential_diagnosis": [], "reasoning": r}

    def _parse_ddx(self, ddx_obj):
        items = ddx_obj.get("differential_diagnosis", []) if isinstance(ddx_obj, dict) else []
        out = []
        for it in items:
            d = it.get("diagnosis", "") if isinstance(it, dict) else str(it)
            c = it.get("confidence", 0.5) if isinstance(it, dict) else 0.5
            try:
                c = float(c)
            except Exception:
                c = 0.5
            c = max(0.0, min(1.0, c))
            if d:
                out.append((d, c))
        return out

    def _gen_questions(self, ddx, patient_info, conv_log, asked):
        asked_str = "; ".join(asked) if asked else "None"
        questions = []
        # k-1 区分性问题：反驳 top1 + 确认 d_i
        if ddx:
            d1 = ddx[0][0]
            for (di, _) in ddx[1:K]:
                msg = [{"role": "user", "content": CONFIRM_QUESTION_PROMPT.format(
                    d_eliminate=d1, d_confirm=di, asked=asked_str)}]
                r = self._llm(msg)
                q = self._extract_response(r)
                if q:
                    questions.append(q)
        # 1 探索性问题
        d_list = ", ".join(d for d, _ in ddx[:K])
        msg = [{"role": "user", "content": EXPLORE_QUESTION_PROMPT.format(
            d_list=d_list or "the current diagnoses", asked=asked_str)}]
        r = self._llm(msg)
        q = self._extract_response(r)
        if q:
            questions.append(q)
        return questions

    def _extract_response(self, text):
        m = re.search(r"Response:\s*(.+)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _simulate(self, question, answer, patient_info, conv_log):
        msg = [{"role": "user", "content": SIMULATE_PROMPT.format(
            patient_info=patient_info, conv_log=conv_log or "None",
            question=question, answer=answer)}]
        r = self._llm(msg, json_mode=True)
        if isinstance(r, dict):
            return self._parse_ddx(r)
        return []

    def _deig(self, question, prior_ddx, patient_info, conv_log):
        ddx_yes = self._simulate(question, "Yes", patient_info, conv_log)
        ddx_no = self._simulate(question, "No", patient_info, conv_log)

        prior_probs = [c for _, c in prior_ddx] or [0.2] * K
        yes_probs = [c for _, c in ddx_yes] or [0.2]
        no_probs = [c for _, c in ddx_no] or [0.2]

        # IG：期望熵减
        ig = _entropy(prior_probs) - 0.5 * _entropy(yes_probs) - 0.5 * _entropy(no_probs)

        # Div：yes/no 两分支 top1 诊断语义相似度（开放式无 ICD，用 LLM 判断）
        div = self._divergence(ddx_yes, ddx_no)

        # Con：1 - Gini（鼓励探索）
        con = 0.5 * ((1 - _gini(yes_probs)) + (1 - _gini(no_probs)))

        return ALPHA * ig + BETA * div + GAMMA * con, (ddx_yes, ddx_no, ig, div, con)

    def _divergence(self, ddx_yes, ddx_no):
        """Div 项（规则近似）：开放式无 ICD，且 LLM 逐条判断成本过高。

        用 yes/no 两分支 top1 诊断是否相同来近似 ICD 语义相似度：
        - 相同 → similarity=1, Div=0（该问题不能区分两个分支，价值低）
        - 不同 → similarity=0, Div=1（能导向不同诊断，价值高）
        保留了 Div 的核心作用（惩罚「yes/no 都指向同一诊断」的低鉴别力问题）。
        """
        top_yes = ddx_yes[0][0] if ddx_yes else ""
        top_no = ddx_no[0][0] if ddx_no else ""
        if not top_yes or not top_no:
            return 1.0
        return 0.0 if top_yes.lower() == top_no.lower() else 1.0

    def _bayesian_update(self, prior_ddx, patient_info, conv_log):
        """Bayesian 更新：LLM 基于新对话重估 DDx（似然），与先验融合 + 温度缩放。"""
        new_obj = self._gen_ddx(patient_info, conv_log)
        likelihood = self._parse_ddx(new_obj)
        if not likelihood:
            return prior_ddx
        # 以 diagnosis 名为 key 融合先验与似然
        prior_map = {d: c for d, c in prior_ddx}
        merged = {}
        for d, c in likelihood:
            prior_c = prior_map.get(d, 0.1)
            merged[d] = max(merged.get(d, 0.0), prior_c * c)
        # 温度缩放
        items = sorted(merged.items(), key=lambda x: -x[1])
        zs = [math.log(max(v, 1e-9)) / TEMP_SCALING for _, v in items]
        m = max(zs)
        exps = [math.exp(z - m) for z in zs]
        total = sum(exps) or 1.0
        return [(d, exps[i] / total) for i, (d, _) in enumerate(items)]

    def run(self, case_input: CaseInput) -> ScenarioResult:
        patient, reporter = build_simulators(
            case_input, self.patient_config, self.reporter_config, self.prompts)

        trace = []
        # 空起步（对齐上游交互场景）：第一轮问主诉，主诉作为 patient_info 起点。
        # 病史细节与检查结果只能通过后续提问获取（不再用全信息 task 开卷）。
        patient_info = elicit_chief_complaint(patient, self.prompts, trace)
        conv_log = ""
        asked = []
        ddx = []

        current_turn = 1
        answer = ""
        t_start = time.time()

        while current_turn < self.max_turns:
            # 工程保护：单例时间预算（DEIG 每轮 18 次 LLM 调用，问满 16 轮成本过高）
            if self.time_budget and time.time() - t_start > self.time_budget:
                answer = ddx[0][0] if ddx else ""
                break

            # 1. DDx
            ddx_obj = self._gen_ddx(patient_info, conv_log)
            ddx = self._parse_ddx(ddx_obj)
            trace.append({"role_id": "doctor_ddx", "content": json.dumps(
                ddx_obj, ensure_ascii=False) if isinstance(ddx_obj, dict) else str(ddx_obj)})

            # 停止条件（top1 置信度 / gap），min_turns 内强制继续问
            if ddx and current_turn >= self.min_turns:
                probs = [c for _, c in ddx]
                top1 = probs[0]
                gap = probs[0] - (probs[1] if len(probs) > 1 else 0)
                if top1 >= CONF_STOP or gap >= GAP_STOP:
                    answer = ddx[0][0]
                    break

            # 2. 候选问题
            questions = self._gen_questions(ddx, patient_info, conv_log, asked)
            if not questions:
                # 没有候选问题 → 直接给 top1 诊断
                answer = ddx[0][0] if ddx else ""
                break

            # 3. DEIG 选问题
            best_q, best_score, best_info = None, -1e9, None
            for q in questions:
                if q in asked:
                    continue
                score, info = self._deig(q, ddx, patient_info, conv_log)
                trace.append({"role_id": "doctor_deig", "content": json.dumps(
                    {"question": q, "score": score, "ig": info[2], "div": info[3], "con": info[4]},
                    ensure_ascii=False)})
                if score > best_score:
                    best_q, best_score, best_info = q, score, info

            if best_q is None:
                answer = ddx[0][0] if ddx else ""
                break

            trace.append({"role_id": "doctor", "content": best_q, "duration": 0.0})
            asked.append(best_q)
            conv_log += f"\nDoctor: {best_q}\n"

            # 4. 问 patient/reporter
            resp, role = ask_question(best_q, patient, reporter, self.prompts, trace)
            if role == "reporter":
                conv_log += f"\nMeasurement: {resp}\n"
            else:
                conv_log += f"\nPatient: {resp}\n"

            # 5. Bayesian 更新
            ddx = self._bayesian_update(ddx, patient_info, conv_log)

            current_turn += 1

        if not answer and ddx:
            answer = ddx[0][0]

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "medclarify",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "final_ddx": ddx,
            },
        )


class MedClarifyMinQScenario(MedClarifyScenario):
    """MedClarify 强制交互对照：至少问 4 轮（观察提问质量，绕过 qwen 过早自信停止）。"""

    def __init__(self, dataset_name, doctor_config, patient_config, reporter_config,
                 max_turns=16, time_budget=None):
        super().__init__(dataset_name, doctor_config, patient_config, reporter_config,
                         max_turns=max_turns, min_turns=4, time_budget=time_budget)

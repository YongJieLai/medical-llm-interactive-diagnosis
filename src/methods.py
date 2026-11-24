"""四个方法 + vanilla 基座（零训练近似实现）。

统一接口：每个方法拿到患者（ScriptPatient），
在 max_turns 内通过"问证据 / 下诊断"的循环逼近诊断，
返回 {diagnosis, correct, turns, trace, asked}。

方法分工（谁决定"问什么"是关键差异）：
- Vanilla    ：LLM 自由选问 + 诊断（基座，也是 CDEG 建图的轨迹来源）
- MedKGI     ：LLM 生成 DDX（候选疾病）→ 算法信息增益选问 → LLM 诊断
- SGUR       ：算法信念分布 + 熵减选问 → LLM 诊断
- KnowGuard  ：证据池 + LLM 判断证据充分性（弃权）+ 针对性追问 → LLM 诊断
"""
import math
from .llm_client import chat
from .ddxplus_loader import condition_evidence_sets, evidence_question

MAX_TURNS = 20


# ---------- 通用工具 ----------

def build_condition_index(conditions):
    """{疾病名: 证据集合}。"""
    return {d: condition_evidence_sets(c) for d, c in conditions.items()}


def info_gain(candidates, evidence, cond_sets):
    """binary 证据的信息增益（对候选疾病集二分的能力）。"""
    n = len(candidates)
    if n <= 1:
        return 0.0
    a = sum(1 for d in candidates if evidence in cond_sets[d])
    if a == 0 or a == n:
        return 0.0  # 无法区分
    def H(k):
        return k * math.log2(k)
    return math.log2(n) - (H(a) + H(n - a)) / n


def format_history(ev, history):
    """把 [(evidence, answer), ...] 转成自然语言历史。"""
    lines = []
    for e, ans in history:
        q = evidence_question(ev, e)
        lines.append(f"Q: {q}  A: {ans}")
    return "\n".join(lines)


# ---------- Prompt ----------

def _condition_list(cond):
    return ", ".join(sorted(cond.keys()))


def diagnosis_prompt(patient, ev, history, cond):
    hist = format_history(ev, history)
    return f"""You are a clinical reasoning assistant doing differential diagnosis.

Patient: {patient.demographics()}
Chief complaint: {patient.initial_utterance()}

Information gathered so far (Q = question asked, A = patient's answer):
{hist if hist else "(none yet)"}

Your task: output the SINGLE most likely diagnosis, chosen from this list of {len(cond)} conditions:
{_condition_list(cond)}

Respond with JSON only:
{{"diagnosis": "<one condition from the list, exact wording>", "reasoning": "<brief>"}}"""


def ddx_prompt(patient, ev, history, cond, k=5):
    hist = format_history(ev, history)
    return f"""You are a clinical reasoning assistant. Based only on the information gathered so far, propose candidate diagnoses.

Patient: {patient.demographics()}
Chief complaint: {patient.initial_utterance()}

Information gathered so far:
{hist if hist else "(none yet)"}

From this list of {len(cond)} conditions:
{_condition_list(cond)}

Output the top-{k} most likely candidate conditions (exact wording from the list).
Respond with JSON only:
{{"candidates": ["cond1", "cond2", ...]}}"""


def vanilla_turn_prompt(patient, ev, history, cond, pool):
    """pool: 剩余可问的问题列表 [(evidence_name, question_en), ...]，只放一部分。"""
    hist = format_history(ev, history)
    pool_lines = "\n".join(f"{i+1}. {q}" for i, (_, q) in enumerate(pool))
    return f"""You are a clinician taking a patient's history. At each turn, EITHER ask ONE question to gather more information, OR make a diagnosis if you already have enough.

Patient: {patient.demographics()}
Chief complaint: {patient.initial_utterance()}

History so far:
{hist if hist else "(none yet)"}

Available questions (choose ONE by its number):
{pool_lines}

Respond with JSON only:
- If asking: {{"action": "ask", "question_index": <integer>}}
- If diagnosing: {{"action": "diagnose", "diagnosis": "<exact condition from the list>"}}

The full condition list: {_condition_list(cond)}"""


# ---------- 方法基类 ----------

class BaseMethod:
    name = "base"

    def __init__(self, client, model, ev, cond):
        self.client = client
        self.model = model
        self.ev = ev
        self.cond = cond
        self.cond_sets = build_condition_index(cond)
        self.cond_names = sorted(cond.keys())
        self.evidence_names = sorted(ev.keys())

    def _diagnose(self, patient, history):
        msg = [{"role": "user", "content": diagnosis_prompt(patient, self.ev, history, self.cond)}]
        out = chat(self.client, self.model, msg, temperature=0.0, json_mode=True)
        if isinstance(out, dict):
            return out.get("diagnosis", "")
        return ""

    def run(self, patient):
        raise NotImplementedError


# ---------- Vanilla ----------

class VanillaDoctor(BaseMethod):
    name = "vanilla"

    def run(self, patient):
        history = []  # [(evidence, answer)]
        trace = []
        # 初始主诉作为首条历史
        patient.initial_utterance()
        for turn in range(MAX_TURNS):
            # 剩余可问证据
            remaining = [(e, evidence_question(self.ev, e))
                         for e in self.evidence_names if e not in {h[0] for h in history}]
            # 池子太大则截断（随机子集，保证多样性 + 控制 token）
            pool = remaining
            msg = [{"role": "user", "content": vanilla_turn_prompt(patient, self.ev, history, self.cond, pool)}]
            out = chat(self.client, self.model, msg, temperature=0.0, json_mode=True)
            if not isinstance(out, dict):
                trace.append({"turn": turn, "action": "error", "raw": str(out)[:200]})
                continue
            action = out.get("action")
            if action == "diagnose":
                diag = out.get("diagnosis", "")
                trace.append({"turn": turn, "action": "diagnose", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
            elif action == "ask":
                idx = out.get("question_index")
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(pool):
                        e = pool[idx][0]
                        ans = patient.answer(e)
                        history.append((e, ans))
                        trace.append({"turn": turn, "action": "ask", "evidence": e,
                                      "question": evidence_question(self.ev, e), "answer": ans})
                        continue
                except (TypeError, ValueError):
                    pass
                # 无效索引：随机问一个
                e = remaining[0][0]
                ans = patient.answer(e)
                history.append((e, ans))
                trace.append({"turn": turn, "action": "ask_fallback", "evidence": e,
                              "question": evidence_question(self.ev, e), "answer": ans})
            else:
                trace.append({"turn": turn, "action": "unknown", "raw": str(out)[:200]})
        # 达到轮数上限：强制诊断
        diag = self._diagnose(patient, history)
        trace.append({"turn": MAX_TURNS, "action": "diagnose_forced", "diagnosis": diag})
        return self._finish(diag, patient, MAX_TURNS, trace, history)

    def _finish(self, diag, patient, turns, trace, history):
        correct = (diag == patient.case["pathology"])
        return {
            "method": self.name, "diagnosis": diag, "correct": correct,
            "turns": turns, "trace": trace,
            "asked": [h[0] for h in history],
        }


# ---------- MedKGI ----------

class MedKGI(BaseMethod):
    name = "medkgi"

    def run(self, patient, ddx_k=5):
        history = []
        trace = []
        prev_candidates = None
        unchanged = 0
        for turn in range(MAX_TURNS):
            # 1) LLM 生成 DDX
            msg = [{"role": "user", "content": ddx_prompt(patient, self.ev, history, self.cond, k=ddx_k)}]
            out = chat(self.client, self.model, msg, temperature=0.0, json_mode=True)
            candidates = []
            if isinstance(out, dict):
                candidates = [c for c in out.get("candidates", []) if c in self.cond_sets]
            # 2) 候选疾病空/不齐时回退：用与已知证据一致的疾病
            if not candidates:
                candidates = self._fallback_candidates(history)
            trace.append({"turn": turn, "action": "ddx", "candidates": candidates})
            # 候选集连续 2 轮不变 -> 停止问诊（原方法的软终止）
            if candidates == prev_candidates:
                unchanged += 1
            else:
                unchanged = 0
            prev_candidates = candidates
            if unchanged >= 2:
                diag = self._diagnose(patient, history)
                trace.append({"turn": turn, "action": "diagnose_converged", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
            # 3) 候选问题集 = 候选疾病关联证据（未问过的）
            asked = {h[0] for h in history}
            cand_ev = set()
            for d in candidates:
                cand_ev |= (self.cond_sets[d] - asked)
            # 4) 信息增益选问
            if cand_ev:
                e = max(cand_ev, key=lambda x: info_gain(candidates, x, self.cond_sets))
                ans = patient.answer(e)
                history.append((e, ans))
                trace.append({"turn": turn, "action": "ask_ig", "evidence": e,
                              "question": evidence_question(self.ev, e), "answer": ans})
                continue
            else:
                # 无更多候选问题，直接诊断
                diag = self._diagnose(patient, history)
                trace.append({"turn": turn, "action": "diagnose", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
        diag = self._diagnose(patient, history)
        trace.append({"turn": MAX_TURNS, "action": "diagnose_forced", "diagnosis": diag})
        return self._finish(diag, patient, MAX_TURNS, trace, history)

    def _fallback_candidates(self, history):
        """无 LLM DDX 时：取与已知阳性证据最一致的疾病 top-5。"""
        pos = {e for e, a in history if a == "Yes." or (a not in ("No.", "Not applicable.", "None of these."))}
        scored = []
        for d, es in self.cond_sets.items():
            score = len(es & pos)
            scored.append((score, d))
        scored.sort(reverse=True)
        return [d for _, d in scored[:5]]

    def _finish(self, diag, patient, turns, trace, history):
        correct = (diag == patient.case["pathology"])
        return {
            "method": self.name, "diagnosis": diag, "correct": correct,
            "turns": turns, "trace": trace,
            "asked": [h[0] for h in history],
        }


# ---------- SGUR ----------

class SGUR(BaseMethod):
    name = "sgur"

    def run(self, patient, beam=5):
        history = []
        trace = []
        prev_top = None
        unchanged = 0
        for turn in range(MAX_TURNS):
            # 1) 用已知证据给所有疾病打分 -> 信念分布（top-beam 作为候选）
            pos = set()
            neg = set()
            for e, a in history:
                if a in ("No.", "Not applicable.", "None of these."):
                    neg.add(e)
                else:
                    pos.add(e)
            scored = []
            for d, es in self.cond_sets.items():
                # 疾病得分：覆盖阳性证据数 - 包含阴性证据数
                score = len(es & pos) - len(es & neg)
                scored.append((score, d))
            scored.sort(reverse=True)
            top_score = scored[0][0] if scored else 0
            # 动态候选集：保留得分接近 top 的疾病（领先超过 1 分则被剪掉），随证据增加自然缩小
            candidates = [d for s, d in scored if s >= top_score - 1]
            if len(candidates) < 1:
                candidates = [d for s, d in scored[:beam]]
            trace.append({"turn": turn, "action": "belief", "candidates": candidates,
                          "top_score": top_score})
            # 信念收敛到单一疾病 -> 停止问诊
            if len(candidates) <= 1:
                diag = self._diagnose(patient, history)
                trace.append({"turn": turn, "action": "diagnose_converged", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
            # 信念 top 连续 2 轮不变 -> 停止问诊
            if candidates == prev_top:
                unchanged += 1
            else:
                unchanged = 0
            prev_top = candidates
            if unchanged >= 2:
                diag = self._diagnose(patient, history)
                trace.append({"turn": turn, "action": "diagnose_converged", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
            # 2) 熵减选问：在候选疾病关联证据里选 IG 最大的
            asked = {h[0] for h in history}
            cand_ev = set()
            for d in candidates:
                cand_ev |= (self.cond_sets[d] - asked)
            if cand_ev:
                e = max(cand_ev, key=lambda x: info_gain(candidates, x, self.cond_sets))
                ans = patient.answer(e)
                history.append((e, ans))
                trace.append({"turn": turn, "action": "ask_ig", "evidence": e,
                              "question": evidence_question(self.ev, e), "answer": ans})
                continue
            else:
                diag = self._diagnose(patient, history)
                trace.append({"turn": turn, "action": "diagnose", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
        diag = self._diagnose(patient, history)
        trace.append({"turn": MAX_TURNS, "action": "diagnose_forced", "diagnosis": diag})
        return self._finish(diag, patient, MAX_TURNS, trace, history)

    def _finish(self, diag, patient, turns, trace, history):
        correct = (diag == patient.case["pathology"])
        return {
            "method": self.name, "diagnosis": diag, "correct": correct,
            "turns": turns, "trace": trace,
            "asked": [h[0] for h in history],
        }


# ---------- KnowGuard ----------

class KnowGuard(BaseMethod):
    name = "knowguard"

    def run(self, patient):
        history = []
        trace = []
        for turn in range(MAX_TURNS):
            asked = {h[0] for h in history}
            remaining = [e for e in self.evidence_names if e not in asked]
            # 每轮让 LLM 判断：证据是否足够诊断？不够则选一个针对性问题
            out = self._abstain_turn(patient, history, remaining)
            if not isinstance(out, dict):
                trace.append({"turn": turn, "action": "error", "raw": str(out)[:200]})
                continue
            action = out.get("action")
            if action == "diagnose":
                diag = out.get("diagnosis", "")
                trace.append({"turn": turn, "action": "diagnose", "diagnosis": diag})
                return self._finish(diag, patient, turn + 1, trace, history)
            # 弃权：选一个针对性问题问
            q = out.get("question", "")
            e = self._match_evidence(q, remaining)
            ans = patient.answer(e)
            history.append((e, ans))
            trace.append({"turn": turn, "action": "abstain_ask", "evidence": e,
                          "question": evidence_question(self.ev, e), "answer": ans,
                          "llm_question": q})
        diag = self._diagnose(patient, history)
        trace.append({"turn": MAX_TURNS, "action": "diagnose_forced", "diagnosis": diag})
        return self._finish(diag, patient, MAX_TURNS, trace, history)

    def _abstain_turn(self, patient, history, remaining):
        hist = format_history(self.ev, history)
        pool_lines = "\n".join(f"{i+1}. {evidence_question(self.ev, e)}" for i, e in enumerate(remaining[:50]))
        prompt = f"""You are a clinician. Decide whether you have enough evidence to make a diagnosis.

Patient: {patient.demographics()}
Chief complaint: {patient.initial_utterance()}

Evidence gathered:
{hist if hist else "(none yet)"}

If evidence is INSUFFICIENT, you must abstain from diagnosing and ask ONE targeted follow-up question instead.
Candidate questions (pick the single most diagnostically valuable, respond with its exact text):
{pool_lines}

Respond with JSON only:
- If evidence sufficient: {{"action": "diagnose", "diagnosis": "<exact condition>"}}
- If insufficient: {{"action": "ask", "question": "<exact text of the one question you choose>"}}

Full condition list: {_condition_list(self.cond)}"""
        return chat(self.client, self.model, [{"role": "user", "content": prompt}],
                    temperature=0.0, json_mode=True)

    def _match_evidence(self, q, remaining):
        """把 LLM 选的问题文本匹配回证据名（精确/子串/首个作为回退）。"""
        for e in remaining:
            if evidence_question(self.ev, e) == q:
                return e
        for e in remaining:
            if q and q.lower() in evidence_question(self.ev, e).lower():
                return e
        return remaining[0] if remaining else self.evidence_names[0]

    def _finish(self, diag, patient, turns, trace, history):
        correct = (diag == patient.case["pathology"])
        return {
            "method": self.name, "diagnosis": diag, "correct": correct,
            "turns": turns, "trace": trace,
            "asked": [h[0] for h in history],
        }


METHODS = {
    "vanilla": VanillaDoctor,
    "medkgi": MedKGI,
    "sgur": SGUR,
    "knowguard": KnowGuard,
}

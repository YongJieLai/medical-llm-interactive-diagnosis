"""冒烟测试：用假 LLM 客户端跑通整条链路，不花 API 额度。

验证点：
1. ScriptPatient 查表回答（阳性给值 / 阴性 No）
2. 四个方法 run() 正常结束、结果字段完整
3. 诊断判对逻辑
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ddxplus_loader import load_evidences, load_conditions
from src.patient import ScriptPatient
from src.methods import VanillaDoctor, MedKGI, SGUR, KnowGuard
from src.runner import is_correct


class FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class FakeChat:
    def __init__(self):
        self.ask_count = 0

    def create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if '"candidates"' in content:
            return FakeResp(json.dumps({"candidates": ["GERD", "Stable angina", "Pericarditis"]}))
        if "Available questions" in content:
            # vanilla turn：前 2 轮 ask，之后 diagnose
            self.ask_count += 1
            if self.ask_count <= 2:
                return FakeResp(json.dumps({"action": "ask", "question_index": 1}))
            return FakeResp(json.dumps({"action": "diagnose", "diagnosis": "GERD"}))
        if "If evidence is INSUFFICIENT" in content:
            self.ask_count += 1
            if self.ask_count <= 2:
                return FakeResp(json.dumps({"action": "ask", "question": "Do you have a cough?"}))
            return FakeResp(json.dumps({"action": "diagnose", "diagnosis": "GERD"}))
        # diagnosis prompt
        return FakeResp(json.dumps({"diagnosis": "GERD", "reasoning": "test"}))


class FakeClient:
    def __init__(self):
        self.chat = type("C", (), {"completions": FakeChat()})()


def main():
    ev = load_evidences()
    cond = load_conditions()
    cases = json.load(open(ROOT / "results" / "cases.json"))["cases"]
    case = cases[0]

    # 1) 患者查表
    p = ScriptPatient(case, ev)
    print("患者:", p.demographics(), "| 主诉:", p.initial_utterance())
    # 找一个真相里的阳性证据和一个阴性证据
    pos_e = list(case["evidence_values"].keys())[0]
    neg_e = [e for e in ev if e not in case["evidence_values"]][0]
    print(f"  问阳性证据 {pos_e}:", p.answer(pos_e))
    print(f"  问阴性证据 {neg_e}:", p.answer(neg_e))

    # 2) 四个方法
    client = FakeClient()
    for cls in [VanillaDoctor, MedKGI, SGUR, KnowGuard]:
        m = cls(client, "fake-model", ev, cond)
        r = m.run(ScriptPatient(case, ev))
        r["correct"] = is_correct(r["diagnosis"], case["pathology"])
        print(f"\n[{cls.name}] diagnosis={r['diagnosis']!r} correct={r['correct']} turns={r['turns']}")
        print(f"   asked={len(r['asked'])} 证据, trace动作数={len(r['trace'])}")
        assert "trace" in r and "asked" in r and "diagnosis" in r
    print("\n冒烟测试通过 ✓")


if __name__ == "__main__":
    main()

import json, glob, re, os
from collections import defaultdict

def extract_action(text):
    if not isinstance(text, str): return "unknown", ""
    m = re.search(r"\[TEST\]\s*([\s\S]*?)(?=\n\[|\Z)", text)
    if m: return "test", m.group(1).strip()
    m = re.search(r"\[QUERY\]\s*([\s\S]*?)(?=\n\[|\Z)", text)
    if m: return "query", m.group(1).strip()
    if "[DIAGNOSIS]" in text: return "diagnosis", ""
    if "[FINISH]" in text: return "finish", ""
    return "unknown", ""

BASE = "benchmarks/eid-benchmark/results/medqa"
METHODS = ["refine_cv_qwen_gap", "refine_cv_qwen_gap_refute", "refine_cv_qwen_gap_refute_rag"]

def load_case_facts():
    facts = {}
    lines = open("benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl").read().strip().split("\n")
    for ln in lines:
        d = json.loads(ln)
        osc = d["OSCE_Examination"]
        tests = osc.get("Test_Results", {})
        pe = osc.get("Physical_Examination_Findings", {})
        facts[d["id"]] = {
            "test_results": tests,
            "pe": pe,
            "exam_facts": d["atomic_facts"]["exam_facts"],
            "truth": osc.get("Correct_Diagnosis", ""),
        }
    return facts

CASE = load_case_facts()

rows = []
for m in METHODS:
    d = f"{BASE}/{m}/qwen-plus/10_turns/record"
    if not os.path.isdir(d): continue
    for fn in sorted(glob.glob(d + "/*.json")):
        try:
            r = json.load(open(fn))
        except: continue
        cid = r["case_id"]
        trace = r["trace"]
        for i, t in enumerate(trace):
            if t.get("role_id") != "doctor": continue
            act, content = extract_action(t.get("content"))
            if act not in ("test", "query"): continue
            # 找紧接着的 reporter/patient 响应
            nxt = trace[i+1] if i+1 < len(trace) else None
            resp_role = None
            resp = None
            if nxt:
                resp_role = nxt.get("role_id")
                resp = nxt.get("content")
            is_ns = isinstance(resp, str) and ("Not specified" in resp or resp.strip().lower()=="not specified")
            is_na = isinstance(resp, str) and ("N/A" in resp and "[RESPONSE]" in resp)
            rows.append({
                "method": m, "case_id": cid, "turn": i,
                "action": act, "content": content[:300],
                "resp_role": resp_role,
                "resp": (resp[:200] if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)[:200]),
                "is_notspecified": is_ns,
                "is_na": is_na,
            })

json.dump(rows, open("/tmp/qa_rows.json", "w"), ensure_ascii=False, indent=1)
# 统计
from collections import Counter
ns = [r for r in rows if r["is_notspecified"]]
na = [r for r in rows if r["is_na"]]
print(f"总 doctor 问询动作: {len(rows)}")
print(f"其中 [TEST]: {sum(1 for r in rows if r['action']=='test')}, [QUERY]: {sum(1 for r in rows if r['action']=='query')}")
print(f"reporter 回 'Not specified': {len(ns)}")
print(f"patient 回 'N/A': {len(na)}")
print()
print("=== 'Not specified' 的 [TEST] 请求（去重）===")
seen = set()
for r in ns:
    if r["action"] != "test": continue
    k = r["content"]
    if k in seen: continue
    seen.add(k)
    print(f"[{r['method']}] {r['case_id']} 轮{r['turn']}: {k}")
json.dump({"all": rows}, open("/tmp/qa_rows.json", "w"), ensure_ascii=False, indent=1)

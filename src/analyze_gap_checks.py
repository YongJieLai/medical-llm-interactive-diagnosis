import json, glob, re, os
from collections import defaultdict, OrderedDict

def extract_action(text):
    if not isinstance(text, str): return "unknown", ""
    # [TEST] 到下一个 [ 标签为止
    m = re.search(r"\[TEST\]\s*([\s\S]*?)(?=\n\[[A-Z]|\Z)", text)
    if m: return "test", m.group(1).strip()
    m = re.search(r"\[QUERY\]\s*([\s\S]*?)(?=\n\[[A-Z]|\Z)", text)
    if m: return "query", m.group(1).strip()
    return "unknown", ""

def is_clean_test(s):
    # 过滤掉明显不是检查请求的（格式错误的 doctor 输出）
    if not s: return False
    bad_pat = re.compile(r"(command is not allowed|No further turns|consultation must end|must wait|feedback states|did not yield|requests\.)", re.I)
    if bad_pat.search(s): return False
    if len(s) > 200: return False  # 太长的往往是推理文本
    # 过滤掉包含 [THOUGHT] 残留的
    if "[THOUGHT]" in s or "THOUGHT:" in s: return False
    return True

lines = open("benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl").read().strip().split("\n")
TRUTH = {}
for ln in lines:
    d = json.loads(ln)
    TRUTH[d["id"]] = d["OSCE_Examination"]["Correct_Diagnosis"]

# 主分析对象：gap_only（60例完整），另附 gap_refute/gap_refute_rag
methods = ["refine_cv_qwen_gap"]
BASE = "benchmarks/eid-benchmark/results/medqa"
bycase = defaultdict(OrderedDict)
for m in methods:
    d = f"{BASE}/{m}/qwen-plus/10_turns/record"
    for fn in sorted(glob.glob(d + "/*.json")):
        try: r = json.load(open(fn))
        except: continue
        cid = r["case_id"]
        trace = r["trace"]
        for i, t in enumerate(trace):
            if t.get("role_id") != "doctor": continue
            act, content = extract_action(t.get("content"))
            if act != "test": continue
            nxt = trace[i+1] if i+1 < len(trace) else None
            if not nxt or nxt.get("role_id") != "reporter": continue
            resp = nxt.get("content")
            if not isinstance(resp, str) or "Not specified" not in resp: continue
            if not is_clean_test(content): continue
            key = content.strip().rstrip('.')
            bycase[cid].setdefault(key, i)

print(f"=== 干净 'Not specified' 检查（gap_only 60例）===")
print(f"涉及 case 数: {len(bycase)}")
total = sum(len(v) for v in bycase.values())
print(f"干净去重检查请求总数: {total}")
print()
out = []
for cid in sorted(bycase):
    truth = TRUTH.get(cid, "?")
    out.append({"case_id": cid, "truth": truth, "tests": list(bycase[cid].keys())})
    print(f"【{cid}】 truth={truth}")
    for k in bycase[cid]:
        print(f"    - {k}")
json.dump(out, open("/tmp/ns_clean.json","w"), ensure_ascii=False, indent=1)

"""关键证据覆盖率分析：区分"诊断关键事实"与"冗余噪音事实"。

背景：eid.metrics.coverage 的 information_coverage_rate 统计的是"采集到的
atomic_facts / 全部 atomic_facts"。但 atomic_facts 里混有大量对诊断无关的
噪音事实（生命体征基线、社会史、否认性陈述、背景信息），会稀释覆盖率。

本脚本用 LLM（DeepSeek）对每个 case 标注：哪些事实是"确诊 ground_truth 诊断
的关键证据"（key），哪些是冗余（noise）。然后计算：
  关键证据覆盖率 = 采集到的 key facts / 全部 key facts
并对比 正确 vs 错误 case 的关键证据覆盖率。

用法：
  python3 src/analyze_key_facts.py --annotate   # 200 case 标注（后台，可并行）
  python3 src/analyze_key_facts.py --report     # 基于标注算关键证据覆盖率
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))

from llm_client import get_llm, chat  # noqa: E402
from eid.metrics.coverage import compute_coverage, extract_fact_ids  # noqa: E402

DATA = REPO / "benchmarks" / "eid-benchmark" / "data" / "agentclinic_medqa_segmented.jsonl"
BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"
KEYF = OUT / "key_facts.json"

INTERACTIVE = ["roleplay", "react", "sc", "refine",
               "refine_cv", "refine_cv_qwen", "refine_cv_no_decisive",
               "refine_cv_qwen_no_decisive", "refine_cv_qwen_check",
               "refine_cv_qwen_gap", "refine_cv_qwen_gap_refute",
               "refine_cv_cover",
               "medclarify", "mediq_abstention", "uot",
               "medclarify_minq", "mediq_minq"]

_SYS = (
    "You are a senior clinician. Given a case's correct diagnosis and its full fact "
    "list (patient history facts + examination/lab facts), identify which facts are "
    "KEY evidence for establishing that diagnosis.\n\n"
    "Criteria:\n"
    "- KEY: decisive history clues, characteristic signs/symptoms, risk factors, and "
    "confirmatory lab/imaging results that establish or strongly point to the diagnosis.\n"
    "- NOISE: routine vital-sign baselines, social history, negative/denial statements, "
    "and background info irrelevant to the diagnosis.\n\n"
    'Output JSON only: {"key_patient": ["<fact numbers>"], "key_exam": ["<fact numbers>"]}\n'
    "The numbers are the leading integers of each fact string."
)


def load_cases() -> list[dict]:
    cases = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            af = d.get("atomic_facts", {}) or {}
            cases.append({
                "case_id": str(d.get("id")),
                "truth": (d.get("OSCE_Examination", {}) or {}).get("Correct_Diagnosis", ""),
                "patient_facts": af.get("patient_facts", []) or [],
                "exam_facts": af.get("exam_facts", []) or [],
            })
    return cases


def annotate_one(case, client, model):
    pf = "\n".join(case["patient_facts"])
    ef = "\n".join(case["exam_facts"])
    prompt = (
        f"Correct diagnosis: {case['truth']}\n\n"
        f"PATIENT HISTORY FACTS:\n{pf}\n\n"
        f"EXAM/LAB FACTS:\n{ef}\n\n"
        "Identify the KEY facts (as fact numbers) for this diagnosis."
    )
    msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}]
    raw = chat(client, model, msgs, temperature=0.0, max_tokens=600, json_mode=True)
    if isinstance(raw, dict):
        return raw
    # 兜底：粗提取 JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"key_patient": [], "key_exam": []}


def annotate(workers=8):
    cases = load_cases()
    client, model = get_llm("deepseek")
    cache = {}
    if KEYF.exists():
        cache = json.loads(KEYF.read_text())

    todo = [c for c in cases if c["case_id"] not in cache]
    print(f"待标注 {len(todo)}/{len(cases)} case")

    def work(c):
        try:
            r = annotate_one(c, client, model)
            return c["case_id"], r
        except Exception as e:
            print(f"[err] {c['case_id']}: {e}", file=sys.stderr)
            return c["case_id"], {"key_patient": [], "key_exam": []}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, r = fut.result()
            cache[cid] = r
            if i % 20 == 0:
                KEYF.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
                print(f"  进度 {i}/{len(todo)}")
    KEYF.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    print(f"标注完成，共 {len(cache)} case -> {KEYF}")


def key_fact_ids(case: dict, keyf: dict) -> tuple[set, set]:
    """返回 (key_patient_ids, key_exam_ids)，用编号集合。"""
    a = keyf.get(case["case_id"], {})
    kp = {str(x) for x in a.get("key_patient", [])}
    ke = {str(x) for x in a.get("key_exam", [])}
    return kp, ke


def collected_key_ids(trace, key_patient, key_exam):
    """采集到的 key fact ids。"""
    cp, ce = set(), set()
    for entry in trace:
        role = entry.get("role_id", "")
        content = entry.get("content", "")
        if role == "patient":
            m = re.search(r"\[REFERENCE\]\s*(.+?)(?=\n\[RESPONSE\]|\n\n|$)", content, re.DOTALL)
            if m and m.group(1).strip().upper() != "N/A":
                cp.update(x for x in re.findall(r"(\d+)\.", m.group(1)) if x in key_patient)
        elif role in {"reporter", "measurement"}:
            ce.update(x for x in re.findall(r"(\d+)\.", content) if x in key_exam)
    return cp, ce


def report():
    keyf = json.loads(KEYF.read_text()) if KEYF.exists() else {}
    cases = {c["case_id"]: c for c in load_cases()}

    print("=" * 100)
    print("关键证据覆盖率（正确组 vs 错误组）—— 只统计确诊关键 facts")
    print("=" * 100)
    print(f"{'方法':22s} {'正确N':>5s} {'正确关键cov':>10s} {'错误N':>5s} {'错误关键cov':>10s} {'差':>7s}")
    print("-" * 65)

    rows = []
    for m in INTERACTIVE:
        pc_file = OUT / f"percase_{m}.json"
        if not pc_file.exists():
            continue
        percase = json.loads(pc_file.read_text())
        # 加载 trace 用于采集判定
        rec_dir = BASE / m / "qwen-plus" / "16_turns" / "record"
        recs = {}
        for fn in sorted(rec_dir.glob("*.json")):
            try:
                r = json.loads(fn.read_text())
                recs[r["case_id"]] = r
            except Exception:
                pass

        corr_cov, inc_cov = [], []
        for cid, v in percase.items():
            if v["correct"] not in (True, False):
                continue
            case = cases.get(cid)
            if not case:
                continue
            kp, ke = key_fact_ids(case, keyf)
            if not kp and not ke:
                continue  # 无 key 标注，跳过
            rec = recs.get(cid)
            if not rec:
                continue
            cp, ce = collected_key_ids(rec["trace"], kp, ke)
            cov = (len(cp) + len(ce)) / (len(kp) + len(ke))
            (corr_cov if v["correct"] else inc_cov).append(cov)

        def avg(lst):
            return sum(lst) / len(lst) if lst else float("nan")

        c = avg(corr_cov)
        i = avg(inc_cov)
        print(f"{m:22s} {len(corr_cov):>5d} {c:>10.1%} {len(inc_cov):>5d} {i:>10.1%} {c-i:>+6.1%}pp")
        rows.append({"mode": m, "correct_n": len(corr_cov), "correct_keycov": c,
                     "incorrect_n": len(inc_cov), "incorrect_keycov": i})

    (OUT / "key_coverage_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n汇总已存: {OUT / 'key_coverage_summary.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--annotate":
        annotate()
    else:
        report()

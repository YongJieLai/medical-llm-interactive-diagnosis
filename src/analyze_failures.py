"""合并重判结果 + 失效指标，定量归类 30 例的失效模式。

产出：
1) 30 例各方法准确率（新 judge）
2) 失效 run 的失效主因归类（方法 × 类型）
3) 失效 case 明细（含证据状态、锚定、正确假设是否出现过）
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/workspace/medical_llm_exp/results")
REJ = json.loads((BASE / "rejudge_analysis.json").read_text())
MET = json.loads((BASE / "failure_metrics.json").read_text())

METHODS = ["roleplay", "refine", "meddxagent", "clinicalagents"]


def classify(m):
    """把一个 run 的 metrics 归类为主因。"""
    if m.get("correct_diagnosis_dropped"):
        return "正确假设被丢弃"
    if m.get("anchored"):
        return "锚定锁死"
    evs = m.get("evidence_status", [])
    if evs:
        n_used = sum(1 for e in evs if e["status"] == "used")
        n_disclosed = sum(1 for e in evs if e["status"] == "disclosed")
        n_notasked = sum(1 for e in evs if e["status"] == "not_asked")
        total = len(evs)
        if n_used == 0 and n_notasked >= total * 0.6:
            return "关键信息未采集"
        if n_disclosed >= 1 and n_used == 0:
            return "证据披露但未利用/读反"
        if n_used >= 1:
            return "证据已用但推理错/粒度不足"
    return "其他"


def main():
    # 30 例判定（10_turns）
    rows = [r for r in REJ["all_rows"] if r["turns"] == "10_turns"]
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        acc[r["method"]][1] += 1
        if r["new_correct"]:
            acc[r["method"]][0] += 1

    print("=" * 60)
    print("30 例准确率（新 judge）")
    print("=" * 60)
    for m in METHODS:
        c, n = acc[m]
        print(f"  {m:16s} {c:2d}/{n:2d} = {c/n*100:.1f}%")

    # 失效分类
    print("\n" + "=" * 60)
    print("失效 run 的失效主因归类（方法 × 类型）")
    print("=" * 60)
    fail_detail = []
    type_counter = defaultdict(Counter)
    for r in rows:
        if r["new_correct"]:
            continue
        key = f"{r['method']}/{r['case_id']}"
        mm = (MET.get(key) or {}).get("metrics") or {}
        cat = classify(mm)
        type_counter[r["method"]][cat] += 1
        fail_detail.append({
            "method": r["method"], "case_id": r["case_id"],
            "ground_truth": r["ground_truth"],
            "prediction": (r["prediction"] or "").split("\n")[0][:70],
            "failure_type": cat,
            "anchored": mm.get("anchored"),
            "dropped": mm.get("correct_diagnosis_dropped"),
            "appeared": mm.get("correct_diagnosis_appeared"),
            "n_notasked": sum(1 for e in mm.get("evidence_status", []) if e["status"] == "not_asked"),
            "n_disclosed": sum(1 for e in mm.get("evidence_status", []) if e["status"] == "disclosed"),
            "n_used": sum(1 for e in mm.get("evidence_status", []) if e["status"] == "used"),
            "initial": (mm.get("initial_hypotheses") or [])[:3],
        })

    types = sorted({c for cc in type_counter.values() for c in cc})
    header = f"{'方法':16s}" + "".join(f"{t:>16s}" for t in types) + f"{'合计':>6s}"
    print(header)
    for m in METHODS:
        cc = type_counter[m]
        row = f"{m:16s}" + "".join(f"{cc.get(t,0):>16d}" for t in types) + f"{sum(cc.values()):>6d}"
        print(row)
    total_cc = Counter()
    for cc in type_counter.values():
        total_cc.update(cc)
    row = f"{'合计':16s}" + "".join(f"{total_cc.get(t,0):>16d}" for t in types) + f"{sum(total_cc.values()):>6d}"
    print(row)

    # 失效明细
    print("\n" + "=" * 60)
    print("失效 case 明细")
    print("=" * 60)
    for d in sorted(fail_detail, key=lambda x: (x["method"], x["case_id"])):
        print(f"[{d['method']:14s}] {d['case_id']:22s} {d['failure_type']:14s} "
              f"anchored={d['anchored']} dropped={d['dropped']} appeared={d['appeared']}")
        print(f"     truth={d['ground_truth']!r}")
        print(f"     pred={d['prediction']!r}")
        print(f"     notasked={d['n_notasked']} disclosed={d['n_disclosed']} used={d['n_used']} "
              f"init={d['initial']}")

    # 保存
    out = {
        "accuracy_30": {m: {"correct": acc[m][0], "total": acc[m][1]} for m in METHODS},
        "failure_type_distribution": {m: dict(type_counter[m]) for m in METHODS},
        "fail_detail": fail_detail,
    }
    (BASE / "failure_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n已写入 {BASE / 'failure_analysis.json'}")


if __name__ == "__main__":
    main()

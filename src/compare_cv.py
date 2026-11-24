"""对比 refine vs refine_cv 的 30 例结果（确定性 judge 同口径）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
CV_REC = EID / "results" / "medqa" / "refine_cv" / "qwen-plus" / "10_turns" / "record"
REJUDGE = Path("/workspace/medical_llm_exp/results/rejudge_analysis.json")


def load_refine_new_correct():
    d = json.loads(REJUDGE.read_text())
    out = {}
    for r in d["all_rows"]:
        if r["method"] == "refine" and r["turns"] == "10_turns":
            out[r["case_id"]] = r
    return out


def main():
    client, model = get_deepseek()
    refine_map = load_refine_new_correct()

    cv_rows = []
    if not CV_REC.is_dir():
        print(f"未找到 refine_cv record: {CV_REC}")
        return
    for fn in sorted(CV_REC.glob("*.json")):
        r = json.loads(fn.read_text())
        cid = r.get("case_id")
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client, model)
        cv_rows.append({
            "case_id": cid, "prediction": pred, "ground_truth": truth,
            "correct": res["correct"], "source": res["source"],
        })

    cv_correct = sum(1 for r in cv_rows if r["correct"] is True)
    refine_correct = sum(1 for cid, r in refine_map.items() if r["new_correct"] is True)
    n = len(cv_rows)

    print("=" * 80)
    print("refine    vs    refine_cv   （30 例，确定性 judge）")
    print("=" * 80)
    print(f"refine    : {refine_correct}/{len(refine_map)} = {refine_correct/len(refine_map)*100:.1f}%")
    print(f"refine_cv : {cv_correct}/{n} = {cv_correct/n*100:.1f}%")
    print(f"变化      : {cv_correct - refine_correct:+d} 例")

    cv_map = {r["case_id"]: r for r in cv_rows}
    repaired, regressed, both_right, both_wrong = [], [], [], []
    for cid in sorted(refine_map):
        ro = refine_map[cid]
        rn = cv_map.get(cid)
        if rn is None:
            continue
        oc, nc = ro["new_correct"], rn["correct"]
        if oc is True and nc is True:
            both_right.append(cid)
        elif oc is True and (nc is False or nc is None):
            regressed.append((cid, ro, rn))
        elif (oc is False or oc is None) and nc is True:
            repaired.append((cid, ro, rn))
        else:
            both_wrong.append(cid)

    print(f"\n修复（refine 错 → cv 对）: {len(repaired)} 例")
    for cid, ro, rn in repaired:
        print(f"  [{cid}] truth={rn['ground_truth']!r}")
        print(f"      refine: {ro['prediction'].split(chr(10))[0][:70]!r}")
        print(f"      cv    : {rn['prediction'].split(chr(10))[0][:70]!r}  src={rn['source']}")

    print(f"\n回归（refine 对 → cv 错）: {len(regressed)} 例")
    for cid, ro, rn in regressed:
        print(f"  [{cid}] truth={rn['ground_truth']!r}")
        print(f"      refine: {ro['prediction'].split(chr(10))[0][:70]!r}")
        print(f"      cv    : {rn['prediction'].split(chr(10))[0][:70]!r}  src={rn['source']}")

    print(f"\n两方法都对: {len(both_right)} 例; 两方法都错: {len(both_wrong)} 例")
    if both_wrong:
        print("  都错 case: " + ", ".join(sorted(both_wrong)))

    out = {
        "refine_correct": refine_correct, "refine_n": len(refine_map),
        "cv_correct": cv_correct, "cv_n": n,
        "repaired": [{"case_id": c, "truth": rn["ground_truth"],
                      "refine_pred": ro["prediction"], "cv_pred": rn["prediction"],
                      "cv_source": rn["source"]} for c, ro, rn in repaired],
        "regressed": [{"case_id": c, "truth": rn["ground_truth"],
                       "refine_pred": ro["prediction"], "cv_pred": rn["prediction"],
                       "cv_source": rn["source"]} for c, ro, rn in regressed],
        "both_wrong": sorted(both_wrong),
        "cv_rows": cv_rows,
    }
    outp = Path("/workspace/medical_llm_exp/results/cv_comparison.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {outp}")


if __name__ == "__main__":
    main()

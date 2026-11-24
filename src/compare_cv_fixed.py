"""对比 refine vs refine_cv（均在新 reporter 环境下），确定性 judge 同口径。

直接读两个 record 目录，各自 judge，逐 case 对比。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
REFINE_REC = EID / "results" / "medqa" / "refine" / "qwen-plus" / "10_turns" / "record"
CV_REC = EID / "results" / "medqa" / "refine_cv" / "qwen-plus" / "10_turns" / "record"


def load_and_judge(rec_dir, client, model):
    rows = {}
    if not rec_dir.is_dir():
        return rows
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            r = json.loads(fn.read_text())
        except Exception:
            continue
        cid = r.get("case_id")
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client, model)
        rows[cid] = {"prediction": pred, "ground_truth": truth,
                     "correct": res["correct"], "source": res["source"]}
    return rows


def main():
    client, model = get_deepseek()
    refine_map = load_and_judge(REFINE_REC, client, model)
    cv_map = load_and_judge(CV_REC, client, model)

    refine_correct = sum(1 for r in refine_map.values() if r["correct"] is True)
    cv_correct = sum(1 for r in cv_map.values() if r["correct"] is True)
    n_refine = len(refine_map)
    n_cv = len(cv_map)

    print("=" * 80)
    print("refine  vs  refine_cv  （新 reporter 环境，30 例，确定性 judge）")
    print("=" * 80)
    print(f"refine    : {refine_correct}/{n_refine} = {refine_correct/n_refine*100:.1f}%")
    print(f"refine_cv : {cv_correct}/{n_cv} = {cv_correct/n_cv*100:.1f}%")

    common = sorted(set(refine_map) & set(cv_map))
    repaired, regressed, both_right, both_wrong = [], [], [], []
    for cid in common:
        ro = refine_map[cid]
        rn = cv_map[cid]
        oc, nc = ro["correct"], rn["correct"]
        if oc is True and nc is True:
            both_right.append(cid)
        elif oc is True and (nc is False or nc is None):
            regressed.append((cid, ro, rn))
        elif (oc is False or oc is None) and nc is True:
            repaired.append((cid, ro, rn))
        else:
            both_wrong.append(cid)

    delta = cv_correct - refine_correct
    print(f"变化      : {delta:+d} 例\n")

    print(f"修复（refine 错 → cv 对）: {len(repaired)} 例")
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
        "refine_correct": refine_correct, "refine_n": n_refine,
        "cv_correct": cv_correct, "cv_n": n_cv,
        "repaired": [{"case_id": c, "truth": rn["ground_truth"],
                      "refine_pred": ro["prediction"], "cv_pred": rn["prediction"],
                      "cv_source": rn["source"]} for c, ro, rn in repaired],
        "regressed": [{"case_id": c, "truth": rn["ground_truth"],
                       "refine_pred": ro["prediction"], "cv_pred": rn["prediction"],
                       "cv_source": rn["source"]} for c, ro, rn in regressed],
        "both_wrong": sorted(both_wrong),
    }
    outp = Path("/workspace/medical_llm_exp/results/cv_comparison_fixed.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {outp}")


if __name__ == "__main__":
    main()

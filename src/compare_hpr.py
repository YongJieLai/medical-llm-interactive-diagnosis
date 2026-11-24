"""对比 refine vs refine_hpr 的 30 例结果（确定性 judge 同口径）。

- refine 的判定复用 results/rejudge_analysis.json（确定性 judge 已判）。
- refine_hpr 用同一个确定性 judge 重新判定。
- 输出：准确率对比、修复清单（refine 错→hpr 对）、回归清单（refine 对→hpr 错）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
HPR_REC = EID / "results" / "medqa" / "refine_hpr" / "qwen-plus" / "10_turns" / "record"
REJUDGE = Path("/workspace/medical_llm_exp/results/rejudge_analysis.json")


def load_refine_new_correct():
    """从 rejudge_analysis.json 提取 refine 的 10_turns 逐 case 新判定。"""
    d = json.loads(REJUDGE.read_text())
    out = {}
    for r in d["all_rows"]:
        if r["method"] == "refine" and r["turns"] == "10_turns":
            out[r["case_id"]] = r
    return out


def main():
    client, model = get_deepseek()
    refine_map = load_refine_new_correct()

    # 判定 refine_hpr
    hpr_rows = []
    if not HPR_REC.is_dir():
        print(f"未找到 refine_hpr record: {HPR_REC}")
        return
    for fn in sorted(HPR_REC.glob("*.json")):
        r = json.loads(fn.read_text())
        cid = r.get("case_id")
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client, model)
        hpr_rows.append({
            "case_id": cid, "prediction": pred, "ground_truth": truth,
            "correct": res["correct"], "source": res["source"],
        })

    hpr_correct = sum(1 for r in hpr_rows if r["correct"] is True)
    refine_correct = sum(1 for cid, r in refine_map.items() if r["new_correct"] is True)
    n = len(hpr_rows)

    print("=" * 80)
    print(f"refine   vs   refine_hpr    （30 例，确定性 judge）")
    print("=" * 80)
    print(f"refine      : {refine_correct}/{len(refine_map)} = {refine_correct/len(refine_map)*100:.1f}%")
    print(f"refine_hpr  : {hpr_correct}/{n} = {hpr_correct/n*100:.1f}%")
    print(f"变化        : {hpr_correct - refine_correct:+d} 例")

    # 逐 case 对比
    hpr_map = {r["case_id"]: r for r in hpr_rows}
    repaired, regressed, both_right, both_wrong, only_hpr = [], [], [], [], []
    for cid in sorted(set(list(refine_map) + list(hpr_map))):
        r_old = refine_map.get(cid)
        r_new = hpr_map.get(cid)
        if r_old is None:
            only_hpr.append((cid, r_new))
            continue
        oc, nc = r_old["new_correct"], (r_new["correct"] if r_new else None)
        if oc is True and nc is True:
            both_right.append(cid)
        elif oc is True and (nc is False or nc is None):
            regressed.append((cid, r_old, r_new))
        elif (oc is False or oc is None) and nc is True:
            repaired.append((cid, r_old, r_new))
        else:
            both_wrong.append(cid)

    print(f"\n修复（refine 错 → hpr 对）: {len(repaired)} 例")
    for cid, ro, rn in repaired:
        print(f"  [{cid}] truth={rn['ground_truth']!r}")
        print(f"      refine: {ro['prediction'].split(chr(10))[0][:70]!r}")
        print(f"      hpr   : {rn['prediction'].split(chr(10))[0][:70]!r}  src={rn['source']}")

    print(f"\n回归（refine 对 → hpr 错）: {len(regressed)} 例")
    for cid, ro, rn in regressed:
        print(f"  [{cid}] truth={rn['ground_truth']!r}")
        print(f"      refine: {ro['prediction'].split(chr(10))[0][:70]!r}")
        print(f"      hpr   : {rn['prediction'].split(chr(10))[0][:70]!r}  src={rn['source']}")

    print(f"\n两方法都对: {len(both_right)} 例; 两方法都错: {len(both_wrong)} 例")
    if both_wrong:
        print("  都错 case: " + ", ".join(sorted(both_wrong)))

    # 存结果
    out = {
        "refine_correct": refine_correct, "refine_n": len(refine_map),
        "hpr_correct": hpr_correct, "hpr_n": n,
        "repaired": [{"case_id": c, "truth": rn["ground_truth"],
                      "refine_pred": ro["prediction"], "hpr_pred": rn["prediction"],
                      "hpr_source": rn["source"]} for c, ro, rn in repaired],
        "regressed": [{"case_id": c, "truth": rn["ground_truth"],
                       "refine_pred": ro["prediction"], "hpr_pred": rn["prediction"],
                       "hpr_source": rn["source"]} for c, ro, rn in regressed],
        "both_wrong": sorted(both_wrong),
        "hpr_rows": hpr_rows,
    }
    outp = Path("/workspace/medical_llm_exp/results/hpr_comparison.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {outp}")


if __name__ == "__main__":
    main()

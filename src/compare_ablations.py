"""对比 refine / refine_cv / refine_cv_qwen / refine_sc 四方法（确定性 judge 同口径）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
METHODS = ["refine", "refine_cv", "refine_cv_qwen", "refine_sc"]


def load_method(method: str, client, model):
    rec_dir = EID / "results" / "medqa" / method / "qwen-plus" / "10_turns" / "record"
    rows = {}
    if not rec_dir.is_dir():
        print(f"[缺] {method}: {rec_dir}")
        return rows
    for fn in sorted(rec_dir.glob("*.json")):
        r = json.loads(fn.read_text())
        cid = r.get("case_id")
        pred = (r.get("prediction") or "").strip()
        truth = (r.get("ground_truth") or "").strip()
        res = judge(pred, truth, client, model)
        rows[cid] = {
            "case_id": cid, "prediction": pred, "ground_truth": truth,
            "correct": res["correct"], "source": res["source"],
        }
    return rows


def main():
    client, model = get_deepseek()
    data = {m: load_method(m, client, model) for m in METHODS}

    # 准确率表
    print("=" * 80)
    print("四方法对比（30 例，确定性 judge，修复 reporter 后）")
    print("=" * 80)
    stats = {}
    for m in METHODS:
        rows = data[m]
        n = len(rows)
        c = sum(1 for r in rows.values() if r["correct"] is True)
        stats[m] = {"correct": c, "n": n}
        pct = f"{c/n*100:.1f}%" if n else "n/a"
        print(f"  {m:16s} : {c}/{n} = {pct}")

    # 以 refine 为基线，算各方法的修复/回归
    base = data["refine"]
    print("\n--- 相对 refine 基线的变化 ---")
    for m in METHODS[1:]:
        rows = data[m]
        if not rows:
            continue
        repaired, regressed, both_wrong = [], [], []
        for cid in sorted(base):
            if cid not in rows:
                continue
            oc = base[cid]["correct"]
            nc = rows[cid]["correct"]
            if oc is True and nc is not True:
                regressed.append(cid)
            elif oc is not True and nc is True:
                repaired.append(cid)
            elif oc is not True and nc is not True:
                both_wrong.append(cid)
        print(f"  {m:16s}: 修复 {len(repaired)} 例, 回归 {len(regressed)} 例, 都错 {len(both_wrong)} 例")

    # 逐 case 明细（跨方法对比）
    all_cids = sorted(base)
    print("\n--- 逐 case 判定（✓对 ✗错 ?不确定）---")
    header = "  " + "case_id".ljust(24)
    for m in METHODS:
        header += m.replace("refine_", "r_").ljust(10)
    print(header)
    for cid in all_cids:
        line = "  " + cid.ljust(24)
        for m in METHODS:
            r = data[m].get(cid)
            if r is None:
                line += "-".ljust(10)
            elif r["correct"] is True:
                line += "✓".ljust(10)
            elif r["correct"] is False:
                line += "✗".ljust(10)
            else:
                line += "?".ljust(10)
        print(line)

    # 落盘
    out = {"stats": stats, "rows": {m: list(v.values()) for m, v in data.items()}}
    outp = Path("/workspace/medical_llm_exp/results/ablation_comparison.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {outp}")


if __name__ == "__main__":
    main()

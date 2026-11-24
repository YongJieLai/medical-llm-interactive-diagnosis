"""用确定性 judge 重判 AgentClinic 场景的所有历史结果，输出旧 vs 新对比。

读取 results/medqa/{method}/qwen-plus/{turns}/record/*.json，
对每条 (prediction, ground_truth) 重新判定，生成：
1) 每方法新旧准确率对比
2) 判定被翻转的 case 清单（噪声修正 / 反向修正）
3) 走 LLM 兜底的 case 清单
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")


def load_records(rec_dir):
    out = []
    if not rec_dir.is_dir():
        return out
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            out.append(json.loads(fn.read_text()))
        except Exception as e:
            print(f"  读取失败 {fn}: {e}")
    return out


def main():
    client, model = get_deepseek()
    methods = ["roleplay", "refine", "meddxagent", "clinicalagents"]

    all_rows = []          # 每条判定的明细
    method_agg = {}        # method -> {turns -> {old, new, n}}

    for m in methods:
        mdir = BASE / m / "qwen-plus"
        if not mdir.is_dir():
            continue
        method_agg[m] = {}
        for turns_dir in sorted(mdir.iterdir()):
            if not turns_dir.is_dir():
                continue
            turns = turns_dir.name
            recs = load_records(turns_dir / "record")
            if not recs:
                continue
            old_c = sum(1 for r in recs if r.get("is_correct"))
            rows = []
            for r in recs:
                cid = r.get("case_id")
                pred = r.get("prediction") or ""
                truth = r.get("ground_truth") or ""
                old = bool(r.get("is_correct"))
                res = judge(pred, truth, client, model)
                new = res["correct"]
                rows.append({
                    "method": m, "turns": turns, "case_id": cid,
                    "prediction": pred, "ground_truth": truth,
                    "old_correct": old, "new_correct": new, "source": res["source"],
                })
                all_rows.append(rows[-1])
            new_c = sum(1 for x in rows if x["new_correct"] is True)
            unk = sum(1 for x in rows if x["new_correct"] is None)
            method_agg[m][turns] = {
                "n": len(rows), "old_correct": old_c, "new_correct": new_c,
                "uncertain": unk,
                "old_acc": old_c / len(rows), "new_acc": new_c / len(rows),
            }

    # 打印对比表
    print("=" * 78)
    print("新旧 judge 准确率对比（10_turns 为主实验，其余为冒烟）")
    print("=" * 78)
    for m in methods:
        for turns in sorted(method_agg.get(m, {})):
            a = method_agg[m][turns]
            print(f"{m:16s} {turns:9s}  n={a['n']:2d}  "
                  f"旧={a['old_acc']*100:5.1f}%  新={a['new_acc']*100:5.1f}%  "
                  f"(旧对{a['old_correct']} 新对{a['new_correct']} 不确定{a['uncertain']})")

    # 翻转明细（只看 10_turns 主实验）
    print("\n" + "=" * 78)
    print("判定被翻转的 case（10_turns）")
    print("=" * 78)
    flipped = [r for r in all_rows if r["turns"] == "10_turns"
               and r["old_correct"] != r["new_correct"] and r["new_correct"] is not None]
    for r in flipped:
        arrow = "错→对(噪声修正)" if (not r["old_correct"] and r["new_correct"]) else "对→错(反向修正)"
        core = r["prediction"].split("\n")[0][:70]
        print(f"  [{r['method']:14s}] {r['case_id']:20s} {arrow:18s} "
              f"truth={r['ground_truth']!r}")
        print(f"       pred={core!r}  src={r['source']}")

    # LLM 兜底清单
    print("\n" + "=" * 78)
    print("走 DeepSeek 兜底的 case（10_turns）")
    print("=" * 78)
    llm_rows = [r for r in all_rows if r["turns"] == "10_turns" and r["source"].startswith("llm")]
    for r in llm_rows:
        core = r["prediction"].split("\n")[0][:70]
        verdict = {True: "CORRECT", False: "INCORRECT", None: "UNKNOWN"}[r["new_correct"]]
        print(f"  [{r['method']:14s}] {r['case_id']:20s} {verdict:9s} "
              f"truth={r['ground_truth']!r}")
        print(f"       pred={core!r}  src={r['source']}")

    # 写明细
    out_path = Path("/workspace/medical_llm_exp/results/rejudge_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method_agg": method_agg,
        "flipped_10turns": flipped,
        "llm_fallback_10turns": llm_rows,
        "all_rows": all_rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {out_path}")


if __name__ == "__main__":
    main()

"""P2 ceiling 分析：oracle DDX-KB 能否救回反证（GapRefuteRAG vs GapRefute vs GapOnly）。

统一用确定性 judge（规则层 + DeepSeek 双判兜底）重判三个变体的前 30 例 trace，
得到同口径的准确率对比，回答：「知识 grounding 能不能把反证从 0.0pp 净贡献
拉到显著为正」。

运行：python3 src/p0_rag_ceiling.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek  # noqa: E402

BASE = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "eid-benchmark" / "results" / "medqa"
)

METHODS = [
    "refine_cv_qwen_gap",            # 只 gap（基线）
    "refine_cv_qwen_gap_refute",     # gap + 反证（无 KB）
    "refine_cv_qwen_gap_refute_rag", # gap + 反证 + oracle KB
]


def load_records(method: str, n: int):
    rec_dir = BASE / method / "qwen-plus" / "10_turns" / "record"
    out = []
    for i in range(n):
        fn = rec_dir / f"agentclinic_medqa_{i}.json"
        if not fn.exists():
            print(f"  [缺] {fn.name}")
            continue
        r = json.loads(fn.read_text())
        out.append(r)
    return out


def judge_method(client, model, method: str, n: int):
    recs = load_records(method, n)
    stats = Counter()
    detail = []
    for r in recs:
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client=client, model=model)
        verdict = res["correct"]
        src = res["source"]
        if verdict is True:
            stats["correct"] += 1
        elif verdict is False:
            stats["incorrect"] += 1
        else:
            stats["uncertain"] += 1
        detail.append({
            "case": r.get("case_id", "?"),
            "pred": pred,
            "truth": truth,
            "correct": verdict,
            "source": src,
        })
    return stats, detail


def main():
    n = 30
    client, model = get_deepseek()
    print(f"oracle KB ceiling 分析（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for m in METHODS:
        stats, detail = judge_method(client, model, m, n)
        total = stats["correct"] + stats["incorrect"] + stats["uncertain"]
        acc = stats["correct"] / total * 100 if total else 0
        decided = stats["correct"] + stats["incorrect"]
        acc_decided = stats["correct"] / decided * 100 if decided else 0
        results[m] = (stats, acc, acc_decided, detail)
        print(f"[{m}]")
        print(f"  correct={stats['correct']}  incorrect={stats['incorrect']}  "
              f"uncertain={stats['uncertain']}  n={total}")
        print(f"  准确率（uncertain 按错）= {acc:.1f}%    （仅 decided）= {acc_decided:.1f}%")
        print()

    # 净贡献
    gap = results["refine_cv_qwen_gap"][1]
    gaprefute = results["refine_cv_qwen_gap_refute"][1]
    gaprefute_rag = results["refine_cv_qwen_gap_refute_rag"][1]
    print("=" * 64)
    print("净贡献（uncertain 按错）：")
    print(f"  反证净贡献 (GapRefute - Gap)        = {gaprefute - gap:+.1f} pp")
    print(f"  RAG 净贡献 (GapRefuteRAG - GapRefute) = {gaprefute_rag - gaprefute:+.1f} pp")
    print(f"  反证+RAG 总贡献 (GapRefuteRAG - Gap) = {gaprefute_rag - gap:+.1f} pp")
    print("=" * 64)

    # 保存明细
    out = {m: results[m][3] for m in METHODS}
    outp = Path(__file__).resolve().parent.parent / "results" / "p0_rag_ceiling.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")

    # case 级：RAG 相对无 RAG 反证改变了什么
    print("\n=== RAG 的影响（GapRefuteRAG vs GapRefute，判定不一致的 case）===")
    d_rag = {d["case"]: d for d in results["refine_cv_qwen_gap_refute_rag"][3]}
    d_no = {d["case"]: d for d in results["refine_cv_qwen_gap_refute"][3]}
    better = worse = 0
    for cid in d_rag:
        a, b = d_rag[cid], d_no[cid]
        if a["correct"] != b["correct"]:
            if a["correct"] and not b["correct"]:
                better += 1
                tag = "RAG 救回"
            else:
                worse += 1
                tag = "RAG 掰歪"
            print(f"  [{cid}] {tag}")
            print(f"      truth     = {a['truth'][:40]!r}")
            print(f"      无RAG pred = {b['pred'][:45]!r} -> {b['correct']}")
            print(f"      RAG  pred = {a['pred'][:45]!r} -> {a['correct']}")
    print(f"\n  救回 {better} 个，掰歪 {worse} 个")


if __name__ == "__main__":
    main()

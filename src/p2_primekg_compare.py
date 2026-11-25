"""P2 不泄题对比：RefineCVQwenGapRefutePrimeKG（HPO 表型 grounding）
vs RefineCVQwenGapRefute（无 RAG baseline）。

同口径确定性 judge（规则层 + DeepSeek 兜底）重判两个变体前 30 例 record，
回答：「不泄题的 HPO disease->phenotype grounding 能否带来净为正的增益」
（对比 oracle KB 的 +6.7pp 知识上界）。

运行：python3 src/p2_primekg_compare.py [N=30]
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

RAG_METHOD = "refine_cv_qwen_gap_refute_primekg"
BASE_METHOD = "refine_cv_qwen_gap_refute"   # 同父类、无 RAG 的 baseline


def load_record(method: str, i: int) -> dict | None:
    fn = BASE / method / "qwen-plus" / "10_turns" / "record" / f"agentclinic_medqa_{i}.json"
    if not fn.exists():
        return None
    return json.loads(fn.read_text())


def judge_method(client, model, method: str, n: int):
    stats = Counter()
    detail = []
    for i in range(n):
        r = load_record(method, i)
        if r is None:
            stats["missing"] += 1
            continue
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client=client, model=model)
        verdict = res["correct"]
        if verdict is True:
            stats["correct"] += 1
        elif verdict is False:
            stats["incorrect"] += 1
        else:
            stats["uncertain"] += 1
        detail.append({
            "case": r.get("case_id", f"medqa_{i}"),
            "pred": pred,
            "truth": truth,
            "correct": verdict,
            "source": res["source"],
        })
    return stats, detail


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    client, model = get_deepseek()
    print(f"P2 PrimeKG/HPO grounding 对比（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for m in (BASE_METHOD, RAG_METHOD):
        stats, detail = judge_method(client, model, m, n)
        total = stats["correct"] + stats["incorrect"] + stats["uncertain"]
        acc = stats["correct"] / total * 100 if total else 0
        decided = stats["correct"] + stats["incorrect"]
        acc_decided = stats["correct"] / decided * 100 if decided else 0
        results[m] = (stats, acc, acc_decided, detail)
        print(f"[{m}]")
        print(f"  correct={stats['correct']}  incorrect={stats['incorrect']}  "
              f"uncertain={stats['uncertain']}  missing={stats.get('missing', 0)}  n={total}")
        print(f"  准确率（uncertain 按错）= {acc:.1f}%    （仅 decided）= {acc_decided:.1f}%")
        print()

    base_acc = results[BASE_METHOD][1]
    rag_acc = results[RAG_METHOD][1]
    print("=" * 64)
    print(f"HPO grounding 净贡献 (primekg - base) = {rag_acc - base_acc:+.1f} pp")
    print(f"  （对比：oracle KB 上界 = +6.7pp）")
    print("=" * 64)

    print("\n=== HPO grounding 的影响（判定不一致的 case）===")
    d_base = {d["case"]: d for d in results[BASE_METHOD][3]}
    d_rag = {d["case"]: d for d in results[RAG_METHOD][3]}
    better = worse = 0
    for cid in d_rag:
        a, b = d_rag[cid], d_base.get(cid)
        if b is None:
            continue
        if a["correct"] != b["correct"]:
            if a["correct"] and not b["correct"]:
                better += 1
                tag = "grounding 救回"
            elif b["correct"] and not a["correct"]:
                worse += 1
                tag = "grounding 掰歪"
            else:
                tag = "grounding 改变(不确定)"
            print(f"  [{cid}] {tag}")
            print(f"      truth     = {a['truth'][:40]!r}")
            print(f"      baseline  = {b['pred'][:45]!r} -> {b['correct']}")
            print(f"      +HPO      = {a['pred'][:45]!r} -> {a['correct']}")
    print(f"\n  救回 {better} 个，掰歪 {worse} 个")

    out = {m: results[m][3] for m in (BASE_METHOD, RAG_METHOD)}
    outp = Path(__file__).resolve().parent.parent / "results" / "p2_primekg.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

"""P0: 完整重判 GapRefute vs GapOnly（规则层 + DeepSeek LLM 兜底），测「反证的净贡献」。

与 p0_refute_net_contribution.py（仅规则层）配套：本脚本在规则层判不出时，
用 DeepSeek 双判 + 多数投票兜底，得到可信的最终准确率与反证净贡献。

运行：python3 src/p0_refute_llm_fallback.py
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
    n = 30  # GapRefute 30 例；GapOnly 取前 30 例对齐
    client, model = get_deepseek()
    print(f"DeepSeek 兜底重判（前 {n} 例），模型 = {model}\n")

    methods = ["refine_cv_qwen_gap_refute", "refine_cv_qwen_gap"]
    results = {}
    for m in methods:
        stats, detail = judge_method(client, model, m, n)
        total = stats["correct"] + stats["incorrect"] + stats["uncertain"]
        acc = stats["correct"] / total * 100 if total else 0
        # 排除 LLM 也判不出的 case 后的口径
        decided = stats["correct"] + stats["incorrect"]
        acc_decided = stats["correct"] / decided * 100 if decided else 0
        results[m] = (stats, acc, acc_decided, detail)
        print(f"[{m}]")
        print(f"  correct={stats['correct']}  incorrect={stats['incorrect']}  "
              f"uncertain={stats['uncertain']}  n={total}")
        print(f"  准确率（uncertain 按错）= {acc:.1f}%")
        print(f"  准确率（仅 decided）= {acc_decided:.1f}%")
        # 来源分布
        src_counter = Counter(d["source"] for d in detail)
        print(f"  来源分布: {dict(src_counter)}")
        print()

    gaprefute_acc = results["refine_cv_qwen_gap_refute"][1]
    gaponly_acc = results["refine_cv_qwen_gap"][1]
    net = gaprefute_acc - gaponly_acc
    net_decided = (
        results["refine_cv_qwen_gap_refute"][2] - results["refine_cv_qwen_gap"][2]
    )
    print("=" * 60)
    print(f"反证净贡献（GapRefute - GapOnly，uncertain 按错）= {net:+.1f} pp")
    print(f"反证净贡献（仅 decided）= {net_decided:+.1f} pp")
    print("=" * 60)

    # 保存完整明细
    out = {m: results[m][3] for m in methods}
    outp = Path(__file__).resolve().parent.parent / "results" / "p0_refute_full_judge.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")

    # 打印两个变体判定不一致的 case（反证到底改变了什么）
    print("\n=== 两变体判定差异 case（反证的影响）===")
    d1 = {d["case"]: d for d in results["refine_cv_qwen_gap_refute"][3]}
    d2 = {d["case"]: d for d in results["refine_cv_qwen_gap"][3]}
    for cid in d1:
        a, b = d1[cid], d2[cid]
        if a["correct"] != b["correct"]:
            arrow = "反证变差" if (b["correct"] and not a["correct"]) else "反证变好"
            print(f"  [{cid}] {arrow}")
            print(f"      GapRefute: pred={a['pred'][:40]!r} -> {a['correct']}")
            print(f"      GapOnly  : pred={b['pred'][:40]!r} -> {b['correct']}")


if __name__ == "__main__":
    main()

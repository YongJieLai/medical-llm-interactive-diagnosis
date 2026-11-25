"""P1 对比：RefineHT（HypothesisTracker 记忆增强）vs baseline（refine_cv_qwen，无 HT）。

统一用确定性 judge（规则层 + DeepSeek 双判兜底）重判两个变体的前 30 例 record，
得到同口径准确率，回答：「跨轮假设记忆（HT）能不能把『正确假设被丢弃（recency bias）』
这一失败模式救回来，产生净为正的增益」。

baseline 选择 refine_cv_qwen（与 refine_ht 同父类 RefineCVQwenScenario、同全开关
claim+gap+refute、同 qwen verifier），唯一差异是「是否注入跨轮假设记忆」。

运行：python3 src/compare_refine_ht.py [N=30]
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

HT_METHOD = "refine_ht"
BASE_METHOD = "refine_cv_qwen"   # 同父类、全开关、无 HT 的 baseline


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
    print(f"P1 HypothesisTracker 对比（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for m in (BASE_METHOD, HT_METHOD):
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
    ht_acc = results[HT_METHOD][1]
    print("=" * 64)
    print(f"HT 净贡献 (refine_ht - refine_cv_qwen) = {ht_acc - base_acc:+.1f} pp")
    print("=" * 64)

    # case 级：HT 相对 baseline 改变了什么
    print("\n=== HT 的影响（判定不一致的 case）===")
    d_base = {d["case"]: d for d in results[BASE_METHOD][3]}
    d_ht = {d["case"]: d for d in results[HT_METHOD][3]}
    better = worse = 0
    for cid in d_ht:
        a, b = d_ht[cid], d_base.get(cid)
        if b is None:
            continue
        if a["correct"] != b["correct"]:
            if a["correct"] and not b["correct"]:
                better += 1
                tag = "HT 救回"
            elif b["correct"] and not a["correct"]:
                worse += 1
                tag = "HT 掰歪"
            else:
                tag = "HT 改变(不确定)"
            print(f"  [{cid}] {tag}")
            print(f"      truth     = {a['truth'][:40]!r}")
            print(f"      baseline  = {b['pred'][:45]!r} -> {b['correct']}")
            print(f"      HT        = {a['pred'][:45]!r} -> {a['correct']}")
    print(f"\n  救回 {better} 个，掰歪 {worse} 个")

    out = {m: results[m][3] for m in (BASE_METHOD, HT_METHOD)}
    outp = Path(__file__).resolve().parent.parent / "results" / "p1_refine_ht.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

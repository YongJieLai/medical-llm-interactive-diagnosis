"""统一对比：四个 refine_cv_qwen 变体前 30 例，同口径确定性 judge。

变体（case_id 完全对齐，均为 qwen-plus / 10_turns / 前 30 例）：
  - gap_refute        : 无 RAG baseline
  - gap_refute_rag    : oracle KB（实验② 同批重跑，坐实「层2 vs oracle」）
  - gap_refute_llmddx : 运行时 LLM 生成 DDx（层 2，+16.7pp）
  - gap_refute_emptyddx : 空 DDx 占位（实验③ 方案 1，区分「格式」vs「内容」）

判读：
  实验② : rag 与 llmddx 的净贡献谁高（同批同口径下）。
  实验③ : 若 emptyddx 净贡献明显为正且接近 llmddx → 增益来自「注入动作/格式」，
          污染排除；若 emptyddx ≈ 0 而 llmddx 为正 → 增益绑定「知识内容」。

运行：python3 src/compare_four_methods.py [N=30]
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
    "refine_cv_qwen_gap_refute",          # baseline
    "refine_cv_qwen_gap_refute_rag",      # oracle KB（实验②）
    "refine_cv_qwen_gap_refute_llmddx",   # 层 2 运行时 LLM
    "refine_cv_qwen_gap_refute_emptyddx", # 空 DDx 占位（实验③）
]
BASE_METHOD = "refine_cv_qwen_gap_refute"


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
        stats["correct" if verdict is True else ("incorrect" if verdict is False else "uncertain")] += 1
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
    print(f"四变体对比（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for m in METHODS:
        stats, detail = judge_method(client, model, m, n)
        total = stats["correct"] + stats["incorrect"] + stats["uncertain"]
        acc = stats["correct"] / total * 100 if total else 0
        results[m] = (stats, acc, detail)
        print(f"[{m}]")
        print(f"  correct={stats['correct']}  incorrect={stats['incorrect']}  "
              f"uncertain={stats['uncertain']}  missing={stats.get('missing', 0)}  n={total}")
        print(f"  准确率（uncertain 按错）= {acc:.1f}%")
        print()

    base_acc = results[BASE_METHOD][1]
    print("=" * 70)
    print(f"{'变体':40s} {'准确率':>8s} {'净贡献':>8s}")
    for m in METHODS:
        acc = results[m][1]
        print(f"{m:40s} {acc:7.1f}% {acc-base_acc:+7.1f} pp")
    print("=" * 70)
    print()
    print("实验② 判读：oracle(rag) 与 层2(llmddx) 的净贡献谁高（同批同口径）")
    print("实验③ 判读：空占位(emptyddx) 净贡献是否明显为正、是否接近 层2(llmddx)")
    print()

    # case 级对照：每个变体 vs baseline 的救回/掰歪
    d_base = {d["case"]: d for d in results[BASE_METHOD][2]}
    for m in METHODS:
        if m == BASE_METHOD:
            continue
        d_m = {d["case"]: d for d in results[m][2]}
        better = worse = changed = 0
        flips = []
        for cid in d_m:
            a, b = d_m[cid], d_base.get(cid)
            if b is None:
                continue
            if a["correct"] != b["correct"]:
                changed += 1
                if a["correct"] and not b["correct"]:
                    better += 1
                    flips.append((cid, "救回", b["pred"], a["pred"]))
                elif b["correct"] and not a["correct"]:
                    worse += 1
                    flips.append((cid, "掰歪", b["pred"], a["pred"]))
                else:
                    flips.append((cid, "不确定", b["pred"], a["pred"]))
        print(f"--- {m} vs baseline ---")
        print(f"  救回 {better} 个，掰歪 {worse} 个（判定翻转 {changed} 个）")
        for cid, tag, bp, ap in flips:
            print(f"    [{cid}] {tag}: baseline={bp[:38]!r} -> 本变体={ap[:38]!r}")
        print()

    out = {m: results[m][2] for m in METHODS}
    outp = Path(__file__).resolve().parent.parent / "results" / "four_methods.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

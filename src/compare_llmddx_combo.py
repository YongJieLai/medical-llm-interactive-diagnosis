"""对比层2 组合/qwen生成 变体 vs baseline/层2/2b，确定性 judge。

本轮同期跑的变体（时间窗一致，最干净）：
  - baseline_rerun : refine_cv_qwen_gap_refute_rerun（gap_refute 同批重跑）
  - llmddxns       : refine_cv_qwen_gap_refute_llmddx_neutralsumm（层2 + 2b 组合）
  - llmddxqwen     : refine_cv_qwen_gap_refute_llmddx_qwengen（层2 DDx 生成换 qwen-plus）

历史变体（参考，注意时间漂移）：
  - baseline_orig  : refine_cv_qwen_gap_refute（早期）
  - llmddx         : refine_cv_qwen_gap_refute_llmddx（层2，DeepSeek 生成 DDx，早期）
  - neutralsumm    : refine_cv_qwen_gap_refute_neutralsumm（2b，P4 期）

核心问题：
  Q1 组合是否叠加：层2 vs 2b vs 层2+2b（llmddxns）
  Q2 qwen 生成是否保住 +16.7pp：层2（DeepSeek 生成）vs llmddxqwen（qwen 生成）
  Q3 baseline 时间漂移多大：baseline_orig vs baseline_rerun

运行：python3 src/compare_llmddx_combo.py [N=30]
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

METHODS = {
    "baseline_orig": "refine_cv_qwen_gap_refute",
    "baseline_rerun": "refine_cv_qwen_gap_refute_rerun",
    "llmddx(层2)": "refine_cv_qwen_gap_refute_llmddx",
    "neutralsumm(2b)": "refine_cv_qwen_gap_refute_neutralsumm",
    "llmddxns(层2+2b)": "refine_cv_qwen_gap_refute_llmddx_neutralsumm",
    "llmddxqwen(层2qgen)": "refine_cv_qwen_gap_refute_llmddx_qwengen",
}


def load_record(method: str, i: int) -> dict | None:
    fn = BASE / method / "qwen-plus" / "10_turns" / "record" / f"agentclinic_medqa_{i}.json"
    if not fn.exists():
        return None
    return json.loads(fn.read_text())


def judge_method(client, model, method: str, n: int):
    stats = Counter()
    detail = {}
    for i in range(n):
        r = load_record(method, i)
        if r is None:
            stats["missing"] += 1
            continue
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client=client, model=model)
        v = res["correct"]
        stats["correct" if v is True else ("incorrect" if v is False else "uncertain")] += 1
        detail[f"agentclinic_medqa_{i}"] = {
            "pred": pred, "truth": truth, "correct": v, "source": res["source"]
        }
    return stats, detail


def acc(s):
    tot = s["correct"] + s["incorrect"] + s["uncertain"]
    return s["correct"] / tot * 100 if tot else 0


def flip_matrix(detail_a, detail_b):
    """返回 (both_right, both_wrong, only_a, only_b)。"""
    both_right = []
    both_wrong = []
    only_a = []
    only_b = []
    for cid in detail_a:
        if cid not in detail_b:
            continue
        a = detail_a[cid]["correct"] is True
        b = detail_b[cid]["correct"] is True
        if a and b:
            both_right.append(cid)
        elif not a and not b:
            both_wrong.append(cid)
        elif a and not b:
            only_a.append(cid)
        else:
            only_b.append(cid)
    return both_right, both_wrong, only_a, only_b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    client, model = get_deepseek()
    print(f"层2 组合/qwen生成 对比（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for label, method in METHODS.items():
        stats, detail = judge_method(client, model, method, n)
        results[label] = {"stats": stats, "detail": detail, "method": method}
        print(
            f"[{label:20s}] correct={stats['correct']:2d} incorrect={stats['incorrect']:2d} "
            f"uncertain={stats['uncertain']:2d} missing={stats['missing']:2d} acc={acc(stats):5.1f}%"
        )

    def A(label):
        return acc(results[label]["stats"])

    # Q3: baseline 时间漂移
    print("\n=== Q3: baseline 时间漂移（orig vs rerun）===")
    br, bw, bo, br2 = flip_matrix(
        results["baseline_orig"]["detail"], results["baseline_rerun"]["detail"]
    )
    print(
        f"  orig={A('baseline_orig'):.1f}% rerun={A('baseline_rerun'):.1f}% "
        f"| 同判 {len(br)+len(bw)}/30（一致率 {(len(br)+len(bw))/30*100:.0f}%），"
        f" orig对rerun错 {len(bo)}，orig错rerun对 {len(br2)}"
    )

    # Q1: 组合是否叠加（层2 vs 2b vs 层2+2b）——历史变体间，标注漂移
    print("\n=== Q1: 组合是否叠加（层2 vs 2b vs 层2+2b）===")
    for a, b in [("llmddx(层2)", "neutralsumm(2b)"), ("neutralsumm(2b)", "llmddxns(层2+2b)")]:
        br, bw, bo, bb = flip_matrix(results[a]["detail"], results[b]["detail"])
        print(
            f"  {a} vs {b}: {A(a):.1f}% vs {A(b):.1f}% | 都对 {len(br)} 都错 {len(bw)} "
            f"只{a}对 {len(bo)} 只{b}对 {len(bb)}"
        )

    # Q2: qwen 生成是否保住（层2 vs 层2qwen）
    print("\n=== Q2: DDx 生成换 qwen 是否保住 +16.7pp（层2 DeepSeek vs 层2 qwen）===")
    br, bw, bo, bb = flip_matrix(
        results["llmddx(层2)"]["detail"], results["llmddxqwen(层2qgen)"]["detail"]
    )
    print(
        f"  llmddx={A('llmddx(层2)'):.1f}% vs llmddxqwen={A('llmddxqwen(层2qgen)'):.1f}% | "
        f"都对 {len(br)} 都错 {len(bw)} 只层2对 {len(bo)} 只层2qwen对 {len(bb)}"
    )
    print(f"    只层2qwen对(救回): {sorted(bb)}")
    print(f"    只层2对(层2qwen掰歪): {sorted(bo)}")

    # 同期三变体干净对比
    print("\n=== 同期三变体干净对比（baseline_rerun 为基线）===")
    base = A("baseline_rerun")
    for label in ["llmddxns(层2+2b)", "llmddxqwen(层2qgen)"]:
        print(f"  {label:20s} 净贡献 = {A(label) - base:+.1f} pp")
    br, bw, bo, bb = flip_matrix(
        results["llmddxns(层2+2b)"]["detail"], results["llmddxqwen(层2qgen)"]["detail"]
    )
    print(f"  层2+2b vs 层2qgen: {A('llmddxns(层2+2b)'):.1f}% vs {A('llmddxqwen(层2qgen)'):.1f}% "
          f"| 都对 {len(br)} 都错 {len(bw)} 只组合对 {len(bo)} 只qgen对 {len(bb)}")

    out = {
        "methods": {k: v["method"] for k, v in results.items()},
        "acc": {k: acc(v["stats"]) for k, v in results.items()},
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    outp = Path(__file__).resolve().parent.parent / "results" / "llmddx_combo_compare.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

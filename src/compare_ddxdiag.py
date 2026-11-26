"""对比方案2 三路（ddxdiag/neutralsumm/joint）vs baseline/doctorddx，确定性 judge。

变体（case_id 对齐，均为 qwen-plus / 10_turns / 前 30 例）：
  - refine_cv_qwen_gap_refute            : baseline（gap_refute，无下游修正）
  - refine_cv_qwen_gap_refute_doctorddx  : P3 DDx 驱动问诊（上游采集）
  - refine_cv_qwen_gap_refute_ddxdiag    : P4 2a diagnostician 两步法（下游防排除）
  - refine_cv_qwen_gap_refute_neutralsumm: P4 2b summarizer 中性转录（下游防裁剪）
  - refine_cv_qwen_gap_refute_joint      : P4 2c 联合（2a+2b）

重点：对「决策排除」9 case 的救回率（方案2 目标根因，35%），以及「证据采集不足」
12 case 是否被下游修正顺带救回。

运行：python3 src/compare_ddxdiag.py [N=30]
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
    "baseline": "refine_cv_qwen_gap_refute",
    "doctorddx": "refine_cv_qwen_gap_refute_doctorddx",
    "ddxdiag(2a)": "refine_cv_qwen_gap_refute_ddxdiag",
    "neutralsumm(2b)": "refine_cv_qwen_gap_refute_neutralsumm",
    "joint(2c)": "refine_cv_qwen_gap_refute_joint",
}

# 根因 case 集（0-based，来自 fromhead-reaudit.md；case_id = agentclinic_medqa_X）
DECISION_EXCLUDE = [15, 18, 25, 35, 36, 44, 52, 54, 59]      # 决策排除 9
GATHER_GAP = [20, 21, 28, 32, 37, 40, 41, 50, 57, 58, 7, 9]  # 证据采集 12


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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    client, model = get_deepseek()
    print(f"方案2 三路 vs baseline/doctorddx（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    results = {}
    for label, method in METHODS.items():
        stats, detail = judge_method(client, model, method, n)
        results[label] = {"stats": stats, "detail": detail, "method": method}
        print(
            f"[{label:16s}] correct={stats['correct']:2d} incorrect={stats['incorrect']:2d} "
            f"uncertain={stats['uncertain']:2d} missing={stats['missing']:2d} acc={acc(stats):5.1f}%"
        )

    print()
    base_acc = acc(results["baseline"]["stats"])
    for label in ["doctorddx", "ddxdiag(2a)", "neutralsumm(2b)", "joint(2c)"]:
        print(f"{label:16s} 净贡献 = {acc(results[label]['stats']) - base_acc:+.1f} pp")

    # case 级翻转（vs baseline）
    base_detail = results["baseline"]["detail"]
    print("\n=== case 级翻转（vs baseline）===")
    flip = {}
    for label in ["doctorddx", "ddxdiag(2a)", "neutralsumm(2b)", "joint(2c)"]:
        detail = results[label]["detail"]
        better, worse = [], []
        for cid in detail:
            a, b = detail[cid], base_detail.get(cid)
            if b is None:
                continue
            if a["correct"] != b["correct"]:
                if a["correct"] and not b["correct"]:
                    better.append(cid)
                elif b["correct"] and not a["correct"]:
                    worse.append(cid)
        flip[label] = {"better": sorted(better), "worse": sorted(worse)}
        print(f"{label:16s} 救回 {len(better)} 掰歪 {len(worse)}")
        print(f"   救回: {sorted(better)}")
        print(f"   掰歪: {sorted(worse)}")

    # 目标根因救回
    print("\n=== 目标根因救回 ===")
    dec_set = {f"agentclinic_medqa_{i}" for i in DECISION_EXCLUDE}
    gap_set = {f"agentclinic_medqa_{i}" for i in GATHER_GAP}
    for label in ["baseline", "doctorddx", "ddxdiag(2a)", "neutralsumm(2b)", "joint(2c)"]:
        detail = results[label]["detail"]
        dec_correct = sorted(c for c in dec_set if detail.get(c, {}).get("correct") is True)
        gap_correct = sorted(c for c in gap_set if detail.get(c, {}).get("correct") is True)
        print(f"{label:16s} 决策排除9例救回 {len(dec_correct)}/9 -> {dec_correct}")
        print(f"{'':16s} 证据采集12例救回 {len(gap_correct)}/12 -> {gap_correct}")

    out = {
        "methods": {k: v["method"] for k, v in results.items()},
        "acc": {k: acc(v["stats"]) for k, v in results.items()},
        "flip": flip,
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    outp = Path(__file__).resolve().parent.parent / "results" / "ddxdiag_compare.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

"""信息覆盖率（Coverage）汇总分析。

背景：AgentClinic 论文提出 coverage 指标 —— 医生在多轮问诊中提取到的相关信息
与 MedQA 病例全部相关信息的比值。论文基线：平均 67%，正确诊断 72% vs 错误 63%。

我们的每条 record 已含 coverage 字段（metrics/coverage.py 计算，口径对齐论文：
patient [REFERENCE] 编号 + reporter 编号 → 收集事实 / atomic_facts 金标准事实）。

本脚本把 coverage 作为一个正式评估维度，做两类分析：
1) 各方法的平均 coverage（overall / patient / exam）；
2) coverage 与准确率的关系（正确 case vs 非正确 case 的 coverage 差异）。

判定口径：rule_judge（True=对；False/None 归为"非正确"，None 单独列不确定数）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")

# 待分析方法（record 目录相对 BASE 的路径）
METHODS = {
    "roleplay(基线)": "roleplay/qwen-plus/10_turns",
    "refine(基线)": "refine/qwen-plus/10_turns",
    "refine_cv_qwen(完整)": "refine_cv_qwen/qwen-plus/10_turns",
    "check_only": "refine_cv_qwen_check/qwen-plus/10_turns",
    "gap_only": "refine_cv_qwen_gap/qwen-plus/10_turns",
    "refute_only(作废)": "refine_cv_qwen_refute/qwen-plus/10_turns",
    "meddxagent(基线)": "meddxagent/qwen-plus/10_turns",
    "meddxagent_gate(插件)": "meddxagent_gate/qwen-plus/10_turns",
    "gap_refute(缺口+反证)": "refine_cv_qwen_gap_refute/qwen-plus/10_turns",
    "gap_refute_rag(+KB)": "refine_cv_qwen_gap_refute_rag/qwen-plus/10_turns",
}


def load_records(method_dir: Path) -> dict[str, dict]:
    out = {}
    record_dir = method_dir / "record"
    if not record_dir.is_dir():
        return out
    for fn in sorted(record_dir.iterdir()):
        if fn.suffix != ".json":
            continue
        try:
            r = json.loads(fn.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "case_id" not in r:
            continue
        out[r["case_id"]] = r
    return out


def analyze(label: str, rel: str) -> dict | None:
    records = load_records(BASE / rel)
    if not records:
        return None

    n = len(records)
    cov_sum = pat_sum = exam_sum = 0.0
    true_cov = true_pat = true_exam = 0.0
    true_n = 0
    false_cov = false_pat = false_exam = 0.0
    false_n = 0
    none_n = 0

    per_case = []
    for cid, r in records.items():
        cov = r.get("coverage", {}) or {}
        overall = cov.get("information_coverage_rate", 0.0)
        pat = cov.get("patient_coverage", 0.0)
        exam = cov.get("exam_coverage", 0.0)

        cov_sum += overall
        pat_sum += pat
        exam_sum += exam

        pred = (r.get("prediction") or "").strip()
        truth = (r.get("ground_truth") or "").strip()
        res = rule_judge(pred, truth)

        per_case.append((cid, overall, res))

        if res is True:
            true_cov += overall
            true_pat += pat
            true_exam += exam
            true_n += 1
        elif res is False:
            false_cov += overall
            false_pat += pat
            false_exam += exam
            false_n += 1
        else:
            none_n += 1
            # None（不确定）归入"非正确"侧统计
            false_cov += overall
            false_pat += pat
            false_exam += exam
            false_n += 1

    return {
        "label": label,
        "n": n,
        "avg_cov": cov_sum / n,
        "avg_pat": pat_sum / n,
        "avg_exam": exam_sum / n,
        "true_n": true_n,
        "true_cov": true_cov / true_n if true_n else 0.0,
        "true_pat": true_pat / true_n if true_n else 0.0,
        "true_exam": true_exam / true_n if true_n else 0.0,
        "false_n": false_n,
        "false_cov": false_cov / false_n if false_n else 0.0,
        "false_pat": false_pat / false_n if false_n else 0.0,
        "false_exam": false_exam / false_n if false_n else 0.0,
        "none_n": none_n,
        "acc": true_n / n if n else 0.0,
        "per_case": per_case,
    }


def main():
    print("=" * 100)
    print("信息覆盖率（Coverage）汇总 —— 对齐 AgentClinic 论文口径")
    print("=" * 100)

    rows = []
    for label, rel in METHODS.items():
        r = analyze(label, rel)
        if r is None:
            print(f"[跳过] {label}: 无数据")
            continue
        rows.append(r)

    # 表1：各方法平均 coverage
    print(f"\n{'方法':28s} {'N':>3s} {'整体cov':>8s} {'患者cov':>8s} {'检查cov':>8s} {'准确率':>7s}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['label']:28s} {r['n']:>3d} "
            f"{r['avg_cov']*100:>7.1f}% {r['avg_pat']*100:>7.1f}% {r['avg_exam']*100:>7.1f}% "
            f"{r['acc']*100:>6.1f}%"
        )

    # 表2：coverage 与准确率关系（正确 vs 非正确）
    print("\n" + "=" * 100)
    print("coverage 与准确率的关系（正确 case vs 非正确 case 的平均 coverage）")
    print("=" * 100)
    print(f"{'方法':28s} {'正确N':>5s} {'正确cov':>8s} {'非正确N':>6s} {'非正确cov':>9s} {'cov差':>7s}")
    print("-" * 75)
    for r in rows:
        gap = r["true_cov"] - r["false_cov"]
        print(
            f"{r['label']:28s} {r['true_n']:>5d} {r['true_cov']*100:>7.1f}% "
            f"{r['false_n']:>6d} {r['false_cov']*100:>8.1f}% {gap*100:>+6.1f}pp"
        )

    # 输出每个 case 明细到 JSON 供后续分析
    out = {
        m["label"]: {
            "avg_coverage": m["avg_cov"],
            "patient_coverage": m["avg_pat"],
            "exam_coverage": m["avg_exam"],
            "accuracy": m["acc"],
            "correct_cov": m["true_cov"],
            "incorrect_cov": m["false_cov"],
            "per_case": {cid: {"coverage": cov, "judge": res} for cid, cov, res in m["per_case"]},
        }
        for m in rows
    }
    out_path = Path("/workspace/medical_llm_exp/results/coverage_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n已落盘 {out_path}")


if __name__ == "__main__":
    main()

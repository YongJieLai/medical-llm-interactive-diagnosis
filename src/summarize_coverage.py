"""从 coverage_analysis/percase_*.json 提炼汇总：覆盖率对比 + 失败根因分类。

失败根因（纯规则，基于 pred 文本 + 覆盖率）：
  1. give_up（放弃诊断）：pred 出现 insufficient / cannot establish / unable to /
     under investigation / inconclusive / not enough 等"不敢下诊断"表述
  2. 采集不足（evidence gap）：综合覆盖率 < 0.25 且非放弃
  3. 决策错（decision error）：综合覆盖率 >= 0.25 且非放弃
  另细分：检查盲区（exam_coverage < 0.15 且 patient_coverage >= 0.3）vs 病史盲区。

用法：python3 src/summarize_coverage.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "coverage_analysis"

# 排除 cot（静态上界，无交互 trace，覆盖率不适用）
INTERACTIVE = ["roleplay", "react", "sc", "refine",
               "refine_cv", "refine_cv_qwen", "refine_cv_no_decisive",
               "refine_cv_qwen_no_decisive", "refine_cv_qwen_check",
               "refine_cv_qwen_gap", "refine_cv_qwen_gap_refute",
               "medclarify", "mediq_abstention", "uot",
               "medclarify_minq", "mediq_minq"]

GIVEUP_PAT = re.compile(
    r"insufficient|unable to|cannot (?:be |)establish|under investigation|"
    r"inconclusive|not enough|no definitive|further (?:evaluation|workup|testing|investigation)|"
    r"unclear|indeterminate|can'?t (?:be |)determin|no specific diagnosis|"
    r"pending|requires? further|undifferentiated|non[- ]specific", re.IGNORECASE
)

NAME = {
    "roleplay": "roleplay(交互基线)",
    "react": "react",
    "sc": "sc(总结诊断)",
    "refine": "refine(论文REFINE)",
    "refine_cv": "CV跨模型四检查",
    "refine_cv_qwen": "CV同模型四检查",
    "refine_cv_no_decisive": "CV跨模型三检查",
    "refine_cv_qwen_no_decisive": "CV同模型三检查",
    "refine_cv_qwen_check": "CV只claim",
    "refine_cv_qwen_gap": "CV只gap",
    "refine_cv_qwen_gap_refute": "CV gap+refute",
    "medclarify": "MedClarify(DEIG)",
    "mediq_abstention": "MEDIQ(abstention)",
    "uot": "UoT(信息增益)",
    "medclarify_minq": "MedClarify(强制问4轮)",
    "mediq_minq": "MEDIQ(强制问4轮)",
}


def load(mode):
    p = OUT / f"percase_{mode}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def classify(v):
    """返回失败根因分类。"""
    pred = (v["pred"] or "").lower()
    if GIVEUP_PAT.search(pred):
        return "give_up"
    if v["information_coverage_rate"] < 0.25:
        return "evidence_gap"
    return "decision_error"


def main():
    rows = []
    for m in INTERACTIVE:
        pc = load(m)
        if not pc:
            continue
        corr = [v for v in pc.values() if v["correct"] is True]
        inc = [v for v in pc.values() if v["correct"] is False]

        def avg(lst, k):
            return sum(v[k] for v in lst) / len(lst) if lst else float("nan")

        # 失败根因分类
        cls_counter = {"give_up": 0, "evidence_gap": 0, "decision_error": 0}
        for v in inc:
            cls_counter[classify(v)] += 1

        rows.append({
            "mode": m,
            "name": NAME.get(m, m),
            "n": len(pc),
            "correct": len(corr),
            "incorrect": len(inc),
            "acc": len(corr) / len(pc) if pc else 0.0,
            "cov_all": avg(list(pc.values()), "information_coverage_rate"),
            "cov_corr": avg(corr, "information_coverage_rate"),
            "cov_inc": avg(inc, "information_coverage_rate"),
            "pat_corr": avg(corr, "patient_coverage"),
            "pat_inc": avg(inc, "patient_coverage"),
            "exam_corr": avg(corr, "exam_coverage"),
            "exam_inc": avg(inc, "exam_coverage"),
            "give_up": cls_counter["give_up"],
            "evidence_gap": cls_counter["evidence_gap"],
            "decision_error": cls_counter["decision_error"],
        })

    # 表1：覆盖率 vs 正确率
    print("=" * 100)
    print("表1  各模式覆盖率对比（正确组 vs 错误组）")
    print("=" * 100)
    print(f"{'方法':22s} {'N':>4s} {'acc':>6s} {'整体cov':>8s} | {'正确cov':>8s} {'错误cov':>8s} {'差':>7s} | {'正确(病/检)':>16s} {'错误(病/检)':>16s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:22s} {r['n']:>4d} {r['acc']:>6.1%} {r['cov_all']:>8.1%} | "
              f"{r['cov_corr']:>8.1%} {r['cov_inc']:>8.1%} {r['cov_corr']-r['cov_inc']:>+6.1%}pp | "
              f"{r['pat_corr']:.0%}/{r['exam_corr']:.0%}{'':>4s} {r['pat_inc']:.0%}/{r['exam_inc']:.0%}")

    # 表2：失败根因分类
    print("\n" + "=" * 100)
    print("表2  错题失败根因分类（占错题比例）")
    print("=" * 100)
    print(f"{'方法':22s} {'错题数':>5s} {'放弃诊断':>8s} {'采集不足':>8s} {'决策错':>8s}")
    print("-" * 60)
    for r in rows:
        n = r["incorrect"] or 1
        print(f"{r['name']:22s} {r['incorrect']:>5d} "
              f"{r['give_up']:>3d}({r['give_up']/n:>5.1%}) "
              f"{r['evidence_gap']:>3d}({r['evidence_gap']/n:>5.1%}) "
              f"{r['decision_error']:>3d}({r['decision_error']/n:>5.1%})")

    # 表3：give_up 型错题的 coverage（放弃诊断时到底采了多少）
    print("\n" + "=" * 100)
    print("表3  '放弃诊断'型错题的覆盖率（门控是否过严的量化证据）")
    print("=" * 100)
    print(f"{'方法':22s} {'放弃数':>5s} {'放弃时整体cov':>10s} {'放弃时检查cov':>10s}")
    print("-" * 55)
    for m in INTERACTIVE:
        pc = load(m)
        if not pc:
            continue
        gv = [v for v in pc.values() if v["correct"] is False and classify(v) == "give_up"]
        if gv:
            cov = sum(v["information_coverage_rate"] for v in gv) / len(gv)
            exam = sum(v["exam_coverage"] for v in gv) / len(gv)
            print(f"{NAME.get(m,m):22s} {len(gv):>5d} {cov:>10.1%} {exam:>10.1%}")

    # 汇总存盘
    (OUT / "coverage_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n汇总已存: {OUT / 'coverage_summary.json'}")


if __name__ == "__main__":
    main()

"""60 例最终确定性数字收口：规则层(已扩充同义词表) + 人工复核口径。

口径（与 HANDOFF §2 发现3 一致）：
- rule_judge 返回 True = 对；
- 返回 False/None = 错（None 经人工复核全部为真实错误，不翻判）。

额外输出：
1) 各方法 60 例 / id0-29 / id30-59 分段准确率；
2) refine vs refine_cv_qwen 及三单组件的 McNemar 成对检验。
"""
import json
import sys
from pathlib import Path
from math import sqrt, erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")

METHODS = {
    "refine(基线)": "refine/qwen-plus/10_turns",
    "refine_cv(跨模型DS)": "refine_cv/qwen-plus/10_turns",
    "refine_cv_qwen(同模型)": "refine_cv_qwen/qwen-plus/10_turns",
    "check(单)": "refine_cv_qwen_check/qwen-plus/10_turns",
    "gap(单)": "refine_cv_qwen_gap/qwen-plus/10_turns",
    "refute(单)": "refine_cv_qwen_refute/qwen-plus/10_turns",
    "qwenmax(强跨模型)": "refine_cv_qwenmax/qwen-plus/10_turns",
    "all_deepseek(同模型DS)": "refine_cv/deepseek/10_turns",
}


def load(method_dir: Path) -> dict:
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
        out[r["case_id"]] = {
            "pred": (r.get("prediction") or "").strip(),
            "truth": (r.get("ground_truth") or "").strip(),
        }
    return out


def is_correct(v) -> bool:
    return rule_judge(v["pred"], v["truth"]) is True


def case_idx(case_id: str) -> int:
    # case_id 形如 agentclinic_medqa_<n>
    return int(case_id.rsplit("_", 1)[-1])


def mcnemar_p(a_correct, b_correct, shared_ids):
    """成对 McNemar 检验（连续性校正），返回 (b, c, chi2, p)。"""
    b = c = 0
    for cid in shared_ids:
        a = a_correct[cid]
        bv = b_correct[cid]
        if a and not bv:
            b += 1
        elif not a and bv:
            c += 1
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # 1-df chi-square 上尾概率 = erfc(sqrt(chi2/2))
    p = erfc(sqrt(chi2 / 2))
    return b, c, chi2, p


def main():
    data = {label: load(BASE / rel) for label, rel in METHODS.items()}

    print("=" * 78)
    print("60 例最终准确率（规则层扩充 + 人工复核：不确定按错）")
    print("=" * 78)
    print(f"{'方法':24s} {'总数':>4s} {'正确':>4s} {'准确率':>8s}")
    print("-" * 78)
    correct = {}
    for label, rows in data.items():
        if not rows:
            continue
        correct[label] = {cid: is_correct(v) for cid, v in rows.items()}
        n = len(rows)
        t = sum(correct[label].values())
        print(f"{label:24s} {n:>4d} {t:>4d} {t/n*100:>7.1f}%")

    print()
    print("=" * 78)
    print("分段对比：id 0–29（旧）vs id 30–59（新）")
    print("=" * 78)
    for label, rows in data.items():
        if not rows:
            continue
        old = {c: v for c, v in correct[label].items() if case_idx(c) < 30}
        new = {c: v for c, v in correct[label].items() if case_idx(c) >= 30}
        if old and new:
            print(f"{label:24s} 旧30例 {sum(old.values()):>2d}/{len(old):>2d} "
                  f"({sum(old.values())/len(old)*100:>5.1f}%)   "
                  f"新30例 {sum(new.values()):>2d}/{len(new):>2d} "
                  f"({sum(new.values())/len(new)*100:>5.1f}%)")

    print()
    print("=" * 78)
    print("成对 McNemar 检验（refine 基线 vs 各方法，60 例共享 case）")
    print("=" * 78)
    base = correct.get("refine(基线)")
    for label in ["refine_cv_qwen(同模型)", "check(单)", "gap(单)", "refute(单)"]:
        if label not in correct:
            continue
        shared = [c for c in base if c in correct[label]]
        b, c, chi2, p = mcnemar_p(base, correct[label], shared)
        sign = "显著" if p < 0.05 else "不显著"
        print(f"  refine vs {label:18s}  b={b:>2d} c={c:>2d}  chi2={chi2:.2f}  p={p:.4f}  ({sign})")

    # 关键结论
    print()
    print("=" * 78)
    print("两个待答问题的结论")
    print("=" * 78)
    rq = correct["refine_cv_qwen(同模型)"]
    rb = correct["refine(基线)"]
    print(f"Q1: refine_cv_qwen 60例 = {sum(rq.values())}/60 = {sum(rq.values())/60*100:.1f}%，"
          f" refine = {sum(rb.values())}/60 = {sum(rb.values())/60*100:.1f}%")
    print(f"    （30例时 73.3% vs 60% → 60例 56.7% vs 51.7%，优势从 +13.3pp 收窄到 +5.0pp）")


if __name__ == "__main__":
    main()

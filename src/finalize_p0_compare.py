"""P0 规则修复三件套的收口对比脚本。

对比四组方法（同为 qwen-plus、10_turns、60 例、确定性 rule_judge 口径）：
- refine(旧基线)        vs refine_v2(REJECT 修复)
- refine_cv_qwen(旧同模型) vs refine_cv_qwen_v2(decisive 门控 + 限轮数<=2)

口径（与 finalize_60.py 一致）：rule_judge 返回 True = 对；False/None = 错
（None 经人工复核全部为真实错误，不翻判）。

额外输出：
1) 各方法 60 例 / id0-29 / id30-59 准确率；
2) 两组修复的逐例 diff（哪些 case 被修好 / 被改坏）；
3) 成对 McNemar 检验。
"""
import json
import sys
from pathlib import Path
from math import sqrt, erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")


def load(rel: str) -> dict:
    record_dir = BASE / rel / "record"
    out = {}
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
    return int(case_id.rsplit("_", 1)[-1])


def mcnemar(a_correct, b_correct, shared_ids):
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
    p = erfc(sqrt(chi2 / 2))
    return b, c, chi2, p


def diff(a_rows, b_rows, a_label, b_label, correct_a, correct_b):
    """列出 b 相对 a 被修好 / 被改坏 / 保持的 case。"""
    shared = sorted(set(a_rows) & set(b_rows), key=case_idx)
    fixed, regressed = [], []
    for cid in shared:
        ca, cb = correct_a[cid], correct_b[cid]
        if not ca and cb:
            fixed.append(cid)
        elif ca and not cb:
            regressed.append(cid)
    print(f"\n  [{a_label} → {b_label}]  修好 {len(fixed)} 例 / 改坏 {len(regressed)} 例")
    if fixed:
        print(f"    ✅ 修好（错→对）:")
        for cid in fixed:
            print(f"       {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"              pred(旧)={a_rows[cid]['pred'][:60]!r}")
            print(f"              pred(新)={b_rows[cid]['pred'][:60]!r}")
    if regressed:
        print(f"    ❌ 改坏（对→错）:")
        for cid in regressed:
            print(f"       {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"              pred(旧)={a_rows[cid]['pred'][:60]!r}")
            print(f"              pred(新)={b_rows[cid]['pred'][:60]!r}")
    return fixed, regressed


def main():
    methods = {
        "refine(旧基线)": "refine/qwen-plus/10_turns",
        "refine_v2(REJECT修复)": "refine_v2/qwen-plus/10_turns",
        "refine_cv_qwen(旧同模型)": "refine_cv_qwen/qwen-plus/10_turns",
        "refine_cv_qwen_v2(门控+限轮)": "refine_cv_qwen_v2/qwen-plus/10_turns",
    }
    rows = {k: load(v) for k, v in methods.items()}
    correct = {k: {cid: is_correct(x) for cid, x in v.items()} for k, v in rows.items()}

    print("=" * 80)
    print("P0 修复三件套 · 60 例最终准确率（确定性 rule_judge，None 算错）")
    print("=" * 80)
    print(f"{'方法':26s} {'总数':>4s} {'正确':>4s} {'准确率':>8s}")
    print("-" * 80)
    for k in methods:
        n = len(rows[k])
        t = sum(correct[k].values())
        print(f"{k:26s} {n:>4d} {t:>4d} {t/n*100:>7.1f}%")

    print("\n" + "=" * 80)
    print("分段对比：id 0–29（旧样本）vs id 30–59（新样本）")
    print("=" * 80)
    for k in methods:
        old = {c: v for c, v in correct[k].items() if case_idx(c) < 30}
        new = {c: v for c, v in correct[k].items() if case_idx(c) >= 30}
        print(f"{k:26s} 旧30例 {sum(old.values()):>2d}/{len(old)} "
              f"({sum(old.values())/len(old)*100:>5.1f}%)   "
              f"新30例 {sum(new.values()):>2d}/{len(new)} "
              f"({sum(new.values())/len(new)*100:>5.1f}%)")

    print("\n" + "=" * 80)
    print("修复一：REJECT 信号修复（refine → refine_v2）逐例 diff")
    print("=" * 80)
    diff(rows["refine(旧基线)"], rows["refine_v2(REJECT修复)"],
         "refine", "refine_v2", correct["refine(旧基线)"], correct["refine_v2(REJECT修复)"])

    print("\n" + "=" * 80)
    print("修复二：decisive 门控 + 限轮数（refine_cv_qwen → refine_cv_qwen_v2）逐例 diff")
    print("=" * 80)
    diff(rows["refine_cv_qwen(旧同模型)"], rows["refine_cv_qwen_v2(门控+限轮)"],
         "refine_cv_qwen", "refine_cv_qwen_v2",
         correct["refine_cv_qwen(旧同模型)"], correct["refine_cv_qwen_v2(门控+限轮)"])

    print("\n" + "=" * 80)
    print("成对 McNemar 检验")
    print("=" * 80)
    pairs = [
        ("refine(旧基线)", "refine_v2(REJECT修复)"),
        ("refine_cv_qwen(旧同模型)", "refine_cv_qwen_v2(门控+限轮)"),
        ("refine_v2(REJECT修复)", "refine_cv_qwen_v2(门控+限轮)"),
    ]
    for a, b in pairs:
        shared = [c for c in correct[a] if c in correct[b]]
        bb, cc, chi2, p = mcnemar(correct[a], correct[b], shared)
        sign = "显著" if p < 0.05 else "不显著"
        print(f"  {a} vs {b:28s}  b={bb:>2d}(旧对→新错) c={cc:>2d}(旧错→新对) "
              f"chi2={chi2:.2f}  p={p:.4f}  ({sign})")

    # 结论摘要
    print("\n" + "=" * 80)
    print("结论摘要")
    print("=" * 80)
    r0 = sum(correct["refine(旧基线)"].values())
    r2 = sum(correct["refine_v2(REJECT修复)"].values())
    c0 = sum(correct["refine_cv_qwen(旧同模型)"].values())
    c2 = sum(correct["refine_cv_qwen_v2(门控+限轮)"].values())
    print(f"REJECT 修复：refine {r0}/60 ({r0/60*100:.1f}%) → refine_v2 {r2}/60 ({r2/60*100:.1f}%)  Δ={r2-r0:+d}")
    print(f"门控+限轮：  refine_cv_qwen {c0}/60 ({c0/60*100:.1f}%) → refine_cv_qwen_v2 {c2}/60 ({c2/60*100:.1f}%)  Δ={c2-c0:+d}")


if __name__ == "__main__":
    main()

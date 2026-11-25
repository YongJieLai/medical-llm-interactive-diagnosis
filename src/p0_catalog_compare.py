"""P1 措辞对齐（医生查表 + 规范目录）对照。

三路对比（同为 qwen-plus、10_turns、60 例、确定性 rule_judge，None 算错）：
- refine_cv_qwen         : closed-world 基线（稀疏 exam_facts，查不到→Not specified）
- refine_cv_qwen_panel   : 离线补全完整检查面板（对齐前，检查名命名混乱）
- refine_cv_qwen_catalog : 完整面板 + 检查名对齐到规范目录 + 医生查表说规范词

核心验证点：
1) 目录对齐后 reporter 'Not specified' 是否进一步下降（措辞对齐消除"问对但词对不上"）；
2) 「面板 → 目录」的增量 = 医生查表的净效果（隔离出"措辞对齐"这一项）；
3) 逐例 diff + McNemar 成对检验。
"""
import json
import sys
from pathlib import Path
from math import sqrt, erfc

sys.path.insert(0, "/workspace/medical_llm_exp/src")
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
            "raw": r,
        }
    return out


def is_correct(v) -> bool:
    return rule_judge(v["pred"], v["truth"]) is True


def case_idx(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[-1])


def count_marker(raw, marker: str) -> int:
    return json.dumps(raw, ensure_ascii=False).count(marker)


def mcnemar(a_correct, b_correct, shared_ids):
    b = c = 0
    for cid in shared_ids:
        a, bv = a_correct[cid], b_correct[cid]
        if a and not bv:
            b += 1
        elif not a and bv:
            c += 1
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = erfc(sqrt(chi2 / 2))
    return b, c, chi2, p


def main():
    methods = {
        "closed-world(基线)": "refine_cv_qwen/qwen-plus/10_turns",
        "完整面板(对齐前)": "refine_cv_qwen_panel/qwen-plus/10_turns",
        "目录对齐(查表)": "refine_cv_qwen_catalog/qwen-plus/10_turns",
    }
    rows = {k: load(v) for k, v in methods.items()}
    correct = {k: {cid: is_correct(x) for cid, x in v.items()} for k, v in rows.items()}

    print("=" * 92)
    print("P1 措辞对齐（医生查表 + 规范目录）· 60 例最终准确率（确定性 rule_judge，None 算错）")
    print("=" * 92)
    print(f"{'方法':26s} {'总数':>4s} {'正确':>4s} {'准确率':>8s}")
    print("-" * 92)
    for k in methods:
        n = len(rows[k])
        if n == 0:
            print(f"{k:26s}  (无结果)")
            continue
        t = sum(correct[k].values())
        print(f"{k:26s} {n:>4d} {t:>4d} {t/n*100:>7.1f}%")

    print("\n分段：id 0–29（旧）vs id 30–59（新）")
    print("-" * 92)
    for k in methods:
        if not rows[k]:
            continue
        old = {c: v for c, v in correct[k].items() if case_idx(c) < 30}
        new = {c: v for c, v in correct[k].items() if case_idx(c) >= 30}
        if old:
            print(f"{k:26s} 旧30例 {sum(old.values()):>2d}/{len(old)} "
                  f"({sum(old.values())/len(old)*100:>5.1f}%)   "
                  f"新30例 {sum(new.values()):>2d}/{len(new)} "
                  f"({sum(new.values())/len(new)*100:>5.1f}%)")

    print("\nreporter 假失败验证（Not specified 计数）")
    print("-" * 92)
    for k in methods:
        if not rows[k]:
            continue
        ns = sum(count_marker(v["raw"], "Not specified") for v in rows[k].values())
        print(f"{k:26s} 'Not specified' 出现 {ns:>4d} 次")

    # 核心：完整面板(对齐前) → 目录对齐(查表)
    if rows["完整面板(对齐前)"] and rows["目录对齐(查表)"]:
        print("\n逐例 diff：完整面板(对齐前) → 目录对齐(查表)")
        print("-" * 92)
        a_rows = rows["完整面板(对齐前)"]
        b_rows = rows["目录对齐(查表)"]
        a_c = correct["完整面板(对齐前)"]
        b_c = correct["目录对齐(查表)"]
        shared = sorted(set(a_rows) & set(b_rows), key=case_idx)
        fixed, regressed = [], []
        for cid in shared:
            ca, cb = a_c[cid], b_c[cid]
            if not ca and cb:
                fixed.append(cid)
            elif ca and not cb:
                regressed.append(cid)
        print(f"  修好 {len(fixed)} 例 / 改坏 {len(regressed)} 例")
        if fixed:
            print("  ✅ 修好（错→对）:")
            for cid in fixed:
                print(f"     {cid}: truth={b_rows[cid]['truth']!r}")
                print(f"           面板={a_rows[cid]['pred'][:50]!r} → 目录={b_rows[cid]['pred'][:50]!r}")
        if regressed:
            print("  ❌ 改坏（对→错）:")
            for cid in regressed:
                print(f"     {cid}: truth={b_rows[cid]['truth']!r}")
                print(f"           面板={a_rows[cid]['pred'][:50]!r} → 目录={b_rows[cid]['pred'][:50]!r}")

        print("\nMcNemar 检验")
        print("-" * 92)
        sh = [c for c in a_c if c in b_c]
        for aname, bname in [("完整面板(对齐前)", "目录对齐(查表)")]:
            bb, cc, chi2, p = mcnemar(correct[aname], correct[bname], sh)
            sign = "显著" if p < 0.05 else "不显著"
            print(f"  {aname} vs {bname}   b={bb:>2d}(对→错) c={cc:>2d}(错→对) "
                  f"chi2={chi2:.2f}  p={p:.4f}  ({sign})")
        # vs 基线
        if rows["closed-world(基线)"]:
            sh2 = [c for c in correct["closed-world(基线)"] if c in b_c]
            bb, cc, chi2, p = mcnemar(correct["closed-world(基线)"], b_c, sh2)
            sign = "显著" if p < 0.05 else "不显著"
            print(f"  closed-world(基线) vs 目录对齐(查表)   b={bb:>2d} c={cc:>2d} "
                  f"chi2={chi2:.2f}  p={p:.4f}  ({sign})")

    print("\n结论摘要")
    print("=" * 92)
    for k in methods:
        if not rows[k]:
            continue
        t = sum(correct[k].values())
        n = len(rows[k])
        print(f"{k:26s} {t}/{n} = {t/n*100:.1f}%")


if __name__ == "__main__":
    main()

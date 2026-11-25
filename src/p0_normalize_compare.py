"""P1 reporter 端归一化（医生自由词 → 查表找规范词）四路对照。

四路（同为 qwen-plus、10_turns、60 例、确定性 rule_judge，None 算错）：
- refine_cv_qwen            : closed-world 基线（稀疏 exam_facts，查不到→Not specified）
- refine_cv_qwen_panel      : 完整检查面板（对齐前，检查名命名混乱）
- refine_cv_qwen_catalog    : 完整面板 + 规范名 + 医生查表说规范词
- refine_cv_qwen_normalize  : 完整面板 + 规范名 + 医生自由词 + reporter 端查表归一化

核心隔离点：
  catalog 与 normalize 的**唯一差别**是"谁做措辞对齐"（医生查表 vs reporter 查表），
  因此两者 diff 隔离出「公开目录给医生」的净效果（此前测得 -1.7pp 负结果）。
  若 normalize ≥ catalog，则证明"措辞对齐的收益可保留，目录带偏的害处可消除"。
"""
import json
import re
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


def count_tests(raw) -> int:
    """统计医生发出的 [TEST] 次数（从 doctor trace 的 content 里数）。"""
    n = 0
    for e in raw.get("trace", []):
        if e.get("role_id") == "doctor":
            n += len(re.findall(r"\[TEST\]", e.get("content", "")))
    return n


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
        "目录对齐(查表·损坏面板)": "refine_cv_qwen_catalog/qwen-plus/10_turns",
        "目录对齐(查表·修复面板)": "refine_cv_qwen_catalog_v2/qwen-plus/10_turns",
        "自由词不归一(修复面板)": "refine_cv_qwen_canonfree/qwen-plus/10_turns",
        "reporter归一化(修复面板)": "refine_cv_qwen_normalize_v2/qwen-plus/10_turns",
    }
    rows = {k: load(v) for k, v in methods.items()}
    correct = {k: {cid: is_correct(x) for cid, x in v.items()} for k, v in rows.items()}

    print("=" * 100)
    print("P1 reporter 端归一化 · 60 例最终准确率（确定性 rule_judge，None 算错）")
    print("=" * 100)
    print(f"{'方法':28s} {'总数':>4s} {'正确':>4s} {'准确率':>9s}")
    print("-" * 100)
    for k in methods:
        n = len(rows[k])
        if n == 0:
            print(f"{k:28s}  (无结果)")
            continue
        t = sum(correct[k].values())
        print(f"{k:28s} {n:>4d} {t:>4d} {t/n*100:>8.1f}%")

    print("\n分段：id 0–29（旧）vs id 30–59（新）")
    print("-" * 100)
    for k in methods:
        if not rows[k]:
            continue
        old = {c: v for c, v in correct[k].items() if case_idx(c) < 30}
        new = {c: v for c, v in correct[k].items() if case_idx(c) >= 30}
        if old:
            old_s = (f"{sum(old.values())}/{len(old)} "
                     f"({sum(old.values())/len(old)*100:.1f}%)") if old else "—"
            new_s = (f"{sum(new.values())}/{len(new)} "
                     f"({sum(new.values())/len(new)*100:.1f}%)") if new else "—"
            print(f"{k:28s} 旧30例 {old_s:>18s}   新30例 {new_s:>18s}")

    print("\n医生问诊行为（[TEST] 次数 / 'Not specified' 次数）")
    print("-" * 100)
    for k in methods:
        if not rows[k]:
            continue
        nt = sum(count_tests(v["raw"]) for v in rows[k].values())
        ns = sum(count_marker(v["raw"], "Not specified") for v in rows[k].values())
        print(f"{k:28s} [TEST] {nt:>4d} 次   'Not specified' {ns:>4d} 次")

    # 核心隔离（修复面板上）：
    #   自由词不归一 → reporter归一化  = 归一化净效果
    #   自由词不归一 → 目录对齐(查表)  = 公开目录净效果
    base_name = "自由词不归一(修复面板)"
    base_rows = rows.get(base_name)
    if base_rows:
        for bname in ["reporter归一化(修复面板)", "目录对齐(查表·修复面板)"]:
            b_rows = rows.get(bname)
            if not b_rows:
                continue
            a_c, b_c = correct[base_name], correct[bname]
            shared = sorted(set(base_rows) & set(b_rows), key=case_idx)
            fixed, regressed = [], []
            for cid in shared:
                ca, cb = a_c[cid], b_c[cid]
                if not ca and cb:
                    fixed.append(cid)
                elif ca and not cb:
                    regressed.append(cid)
            print(f"\n逐例 diff：{base_name} → {bname}")
            print("-" * 100)
            print(f"  修好 {len(fixed)} 例 / 改坏 {len(regressed)} 例")
            for cid in fixed:
                print(f"  ✅ {cid}: truth={b_rows[cid]['truth']!r}")
                print(f"       前={base_rows[cid]['pred'][:50]!r} → 后={b_rows[cid]['pred'][:50]!r}")
            for cid in regressed:
                print(f"  ❌ {cid}: truth={b_rows[cid]['truth']!r}")
                print(f"       前={base_rows[cid]['pred'][:50]!r} → 后={b_rows[cid]['pred'][:50]!r}")

    print("\nMcNemar 成对检验")
    print("-" * 100)
    sh = [c for c in correct[base_name] if c in correct[base_name]]
    pairs = [
        ("完整面板(对齐前)", "自由词不归一(修复面板)"),
        ("自由词不归一(修复面板)", "reporter归一化(修复面板)"),
        ("自由词不归一(修复面板)", "目录对齐(查表·修复面板)"),
        ("reporter归一化(修复面板)", "目录对齐(查表·修复面板)"),
        ("closed-world(基线)", "reporter归一化(修复面板)"),
    ]
    for aname, bname in pairs:
        if not rows.get(aname) or not rows.get(bname):
            continue
        shp = [c for c in correct[aname] if c in correct[bname]]
        bb, cc, chi2, p = mcnemar(correct[aname], correct[bname], shp)
        sign = "显著" if p < 0.05 else "不显著"
        print(f"  {aname} vs {bname}   b={bb:>2d}(对→错) c={cc:>2d}(错→对) "
              f"chi2={chi2:.2f}  p={p:.4f}  ({sign})")

    print("\n结论摘要")
    print("=" * 100)
    for k in methods:
        if not rows[k]:
            continue
        t = sum(correct[k].values())
        n = len(rows[k])
        print(f"{k:28s} {t}/{n} = {t/n*100:.1f}%")


if __name__ == "__main__":
    main()

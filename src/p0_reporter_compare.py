"""P0 reporter 口径三态对比（closed / open / availability）。

对比（同为 qwen-plus、10_turns、60 例、确定性 rule_judge，None 算错）：
- refine_cv_qwen              : closed-world 基线（查不到→Not specified）
- refine_cv_qwen_openworld    : open-world（方案 A，查不到→Normal，已验证有害）
- refine_cv_qwen_availability : availability（方案 B，查不到→Not available + 可请求目录）

额外统计：
1) reporter 返回 'Not specified' / 'Normal' / 'Not available' 次数（验证口径切换）；
2) 逐例 diff（availability vs closed-world 基线），重点看假阴性是否被抑制；
3) McNemar 成对检验。
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
        "open-world(方案A)": "refine_cv_qwen_openworld/qwen-plus/10_turns",
        "availability(方案B)": "refine_cv_qwen_availability/qwen-plus/10_turns",
    }
    rows = {k: load(v) for k, v in methods.items()}
    correct = {k: {cid: is_correct(x) for cid, x in v.items()} for k, v in rows.items()}

    print("=" * 86)
    print("P0 reporter 口径三态 · 60 例最终准确率（确定性 rule_judge，None 算错）")
    print("=" * 86)
    print(f"{'方法':26s} {'总数':>4s} {'正确':>4s} {'准确率':>8s}")
    print("-" * 86)
    for k in methods:
        n = len(rows[k])
        t = sum(correct[k].values())
        print(f"{k:26s} {n:>4d} {t:>4d} {t/n*100:>7.1f}%")

    print("\n分段：id 0–29（旧）vs id 30–59（新）")
    print("-" * 86)
    for k in methods:
        old = {c: v for c, v in correct[k].items() if case_idx(c) < 30}
        new = {c: v for c, v in correct[k].items() if case_idx(c) >= 30}
        print(f"{k:26s} 旧30例 {sum(old.values()):>2d}/{len(old)} "
              f"({sum(old.values())/len(old)*100:>5.1f}%)   "
              f"新30例 {sum(new.values()):>2d}/{len(new)} "
              f"({sum(new.values())/len(new)*100:>5.1f}%)")

    print("\nreporter 口径切换验证（标记计数）")
    print("-" * 86)
    print(f"{'方法':26s} {'Not specified':>14s} {'Normal':>8s} {'Not available':>14s}")
    for k in methods:
        ns = sum(count_marker(v["raw"], "Not specified") for v in rows[k].values())
        nrm = sum(count_marker(v["raw"], "Normal") for v in rows[k].values())
        na = sum(count_marker(v["raw"], "Not available") for v in rows[k].values())
        print(f"{k:26s} {ns:>14d} {nrm:>8d} {na:>14d}")

    # 核心对比：availability vs closed-world 基线
    print("\n逐例 diff：closed-world(基线) → availability(方案B)")
    print("-" * 86)
    a_rows, b_rows = rows["closed-world(基线)"], rows["availability(方案B)"]
    a_c, b_c = correct["closed-world(基线)"], correct["availability(方案B)"]
    shared = sorted(set(a_rows) & set(b_rows), key=case_idx)
    fixed, regressed = [], []
    for cid in shared:
        ca, cb = a_c[cid], b_c[cid]
        if not ca and cb:
            fixed.append(cid)
        elif ca and not cb:
            regressed.append(cid)
    print(f"  修好 {len(fixed)} 例 / 改坏 {len(regressed)} 例（改坏=假阴性候选）")
    if fixed:
        print("  ✅ 修好（错→对）:")
        for cid in fixed:
            print(f"     {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"            旧={a_rows[cid]['pred'][:50]!r} → 新={b_rows[cid]['pred'][:50]!r}")
    if regressed:
        print("  ❌ 改坏（对→错，假阴性候选）:")
        for cid in regressed:
            print(f"     {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"            旧={a_rows[cid]['pred'][:50]!r} → 新={b_rows[cid]['pred'][:50]!r}")

    print("\nMcNemar 检验（vs closed-world 基线）")
    print("-" * 86)
    base = correct["closed-world(基线)"]
    for label in ["open-world(方案A)", "availability(方案B)"]:
        sh = [c for c in base if c in correct[label]]
        bb, cc, chi2, p = mcnemar(base, correct[label], sh)
        sign = "显著" if p < 0.05 else "不显著"
        print(f"  baseline vs {label:20s} b={bb:>2d}(对→错) c={cc:>2d}(错→对) "
              f"chi2={chi2:.2f}  p={p:.4f}  ({sign})")

    print("\n结论摘要")
    print("=" * 86)
    for k in methods:
        t = sum(correct[k].values())
        print(f"{k:26s} {t}/60 = {t/60*100:.1f}%")


if __name__ == "__main__":
    main()

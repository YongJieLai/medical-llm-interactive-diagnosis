"""P1 后续（SDBench 式）对照：诊断中性合成结果 vs Not specified。

两路（同为 qwen-plus、10_turns、60 例、修复后的 canonical 面板、医生自由词、不归一化）：
- refine_cv_qwen_canonfree : 查不到 → 'Not specified'（closed-world）
- refine_cv_qwen_synth     : 查不到 → 诊断中性合成结果（SDBench 式，配 ledger）

唯一变量：reporter 对"查不到"的检查返回什么。隔离出 SDBench 式合成的净效果：
  合成消除"Not specified 泄漏"的收益 vs 合成结果的"误导/泄露"害处，哪个更大。

运行：python3 src/p0_synth_compare.py
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
    n = 0
    for e in raw.get("trace", []):
        if e.get("role_id") == "doctor":
            n += len(re.findall(r"\[TEST\]", e.get("content", "")))
    return n


def synthetic_findings(raw) -> list:
    md = raw.get("metadata", {})
    return md.get("synthetic_findings", []) or []


# 启发式审计：合成结果里的"误导性阴性" / "泄露性阳性"信号词
NEGATIVE_HINTS = ("not detected", "negative", "normal", "absent", "no evidence",
                  "unremarkable", "within normal", "nonreactive", "non-reactive")
POSITIVE_HINTS = ("positive", "elevated", "increased", "present", "reactive",
                  "consistent with", "diagnostic")


def audit_synth(raw) -> dict:
    """对每个合成结果做粗粒度审计：含阴性/阳性信号词计数。"""
    neg = pos = neutral = 0
    for s in synthetic_findings(raw):
        c = (s.get("content") or "").lower()
        if any(w in c for w in NEGATIVE_HINTS):
            neg += 1
        elif any(w in c for w in POSITIVE_HINTS):
            pos += 1
        else:
            neutral += 1
    return {"neg": neg, "pos": pos, "neutral": neutral, "total": len(synthetic_findings(raw))}


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
        "Not specified(对照)": "refine_cv_qwen_canonfree/qwen-plus/10_turns",
        "诊断中性合成(本次)": "refine_cv_qwen_synth/qwen-plus/10_turns",
    }
    rows = {k: load(v) for k, v in methods.items()}
    correct = {k: {cid: is_correct(x) for cid, x in v.items()} for k, v in rows.items()}

    print("=" * 100)
    print("SDBench 式诊断中性合成 · 60 例准确率（确定性 rule_judge，None 算错）")
    print("=" * 100)
    print(f"{'方法':24s} {'总数':>4s} {'正确':>4s} {'准确率':>9s}")
    print("-" * 100)
    for k in methods:
        n = len(rows[k])
        if n == 0:
            print(f"{k:24s}  (无结果)")
            continue
        t = sum(correct[k].values())
        print(f"{k:24s} {n:>4d} {t:>4d} {t/n*100:>8.1f}%")

    print("\n医生问诊行为（[TEST] 次数 / 'Not specified' 次数 / 合成次数）")
    print("-" * 100)
    for k in methods:
        if not rows[k]:
            continue
        nt = sum(count_tests(v["raw"]) for v in rows[k].values())
        ns = sum(count_marker(v["raw"], "Not specified") for v in rows[k].values())
        sy = sum(len(synthetic_findings(v["raw"])) for v in rows[k].values())
        print(f"{k:24s} [TEST] {nt:>4d}   'Not specified' {ns:>4d}   合成 {sy:>4d}")

    print("\n合成结果质量粗审计（信号词计数：阴性/阳性/中性）")
    print("-" * 100)
    if rows.get("诊断中性合成(本次)"):
        tot = {"neg": 0, "pos": 0, "neutral": 0, "total": 0}
        for v in rows["诊断中性合成(本次)"].values():
            a = audit_synth(v["raw"])
            for kk in tot:
                tot[kk] += a[kk]
        print(f"  合成总数 {tot['total']}  含阴性词 {tot['neg']}  含阳性词 {tot['pos']}  中性 {tot['neutral']}")

    # 逐例 diff
    a_name, b_name = "Not specified(对照)", "诊断中性合成(本次)"
    a_rows, b_rows = rows.get(a_name), rows.get(b_name)
    if a_rows and b_rows:
        shared = sorted(set(a_rows) & set(b_rows), key=case_idx)
        a_c, b_c = correct[a_name], correct[b_name]
        fixed, regressed = [], []
        for cid in shared:
            ca, cb = a_c[cid], b_c[cid]
            if not ca and cb:
                fixed.append(cid)
            elif ca and not cb:
                regressed.append(cid)
        print(f"\n逐例 diff：{a_name} → {b_name}")
        print("-" * 100)
        print(f"  修好 {len(fixed)} 例 / 改坏 {len(regressed)} 例")
        for cid in fixed:
            print(f"  ✅ {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"       前={a_rows[cid]['pred'][:48]!r} → 后={b_rows[cid]['pred'][:48]!r}")
        for cid in regressed:
            print(f"  ❌ {cid}: truth={b_rows[cid]['truth']!r}")
            print(f"       前={a_rows[cid]['pred'][:48]!r} → 后={b_rows[cid]['pred'][:48]!r}")

        shp = [c for c in a_c if c in b_c]
        bb, cc, chi2, p = mcnemar(a_c, b_c, shp)
        sign = "显著" if p < 0.05 else "不显著"
        print(f"\nMcNemar: b={bb}(对→错) c={cc}(错→对) chi2={chi2:.2f} p={p:.4f} ({sign})")

    print("\n结论摘要")
    print("=" * 100)
    for k in methods:
        if not rows[k]:
            continue
        t = sum(correct[k].values())
        n = len(rows[k])
        print(f"{k:24s} {t}/{n} = {t/n*100:.1f}%")


if __name__ == "__main__":
    main()

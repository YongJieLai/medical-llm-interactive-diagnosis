"""P2 RAG ablation analysis: full pipeline WITHOUT vs WITH external DDX knowledge.

Compares refine_cv_qwen (baseline) against refine_cv_qwen_rag (RAG-injected) on the
same 60 medqa cases, using the deterministic judge. Focus is the REFUTATION
component: how many cases it fires on, and whether the injected knowledge makes its
counter-evidence / alternative clinically correct instead of misleading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge  # noqa: E402

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")
MODES = {
    "baseline": "refine_cv_qwen/qwen-plus/10_turns",
    "rag": "refine_cv_qwen_rag/qwen-plus/10_turns",
}


def load_cases(rel: str) -> dict:
    out = {}
    rec = BASE / rel / "record"
    if not rec.exists():
        return out
    for fn in sorted(rec.iterdir()):
        if fn.suffix != ".json":
            continue
        try:
            r = json.loads(fn.read_text())
        except Exception:
            continue
        cid = r.get("case_id")
        if not cid:
            continue
        # collect refutation events from verifier trace
        refutes = []
        for t in r.get("trace", []):
            if t.get("role_id") == "verifier" and isinstance(t.get("content"), dict):
                refut = t["content"].get("refutation", {}) or {}
                if refut.get("triggered"):
                    refutes.append(refut)
        out[cid] = {
            "pred": (r.get("prediction") or "").strip(),
            "truth": (r.get("ground_truth") or "").strip(),
            "refutes": refutes,
        }
    return out


def main() -> None:
    base = load_cases(MODES["baseline"])
    rag = load_cases(MODES["rag"])
    if not base or not rag:
        print("缺少数据：baseline=", len(base), "rag=", len(rag))
        return
    ids = sorted(set(base) & set(rag), key=lambda c: int(c.rsplit("_", 1)[-1]))

    def ok(c):
        return rule_judge(c["pred"], c["truth"]) is True

    print("=" * 70)
    print("P2 RAG 对比：完整版 无RAG vs 有RAG（60 例确定性 judge）")
    print("=" * 70)
    n_base = sum(1 for c in ids if ok(base[c]))
    n_rag = sum(1 for c in ids if ok(rag[c]))
    print(f"基线 refine_cv_qwen    : {n_base}/{len(ids)} = {n_base/len(ids)*100:.1f}%")
    print(f"RAG   refine_cv_qwen_rag: {n_rag}/{len(ids)} = {n_rag/len(ids)*100:.1f}%")
    print()

    # refutation firing stats
    n_base_trig = sum(1 for c in ids if base[c]["refutes"])
    n_rag_trig = sum(1 for c in ids if rag[c]["refutes"])
    print(f"refutation 触发 case 数: 基线={n_base_trig}  RAG={n_rag_trig}")
    trig_base_ok = sum(1 for c in ids if base[c]["refutes"] and ok(base[c]))
    trig_rag_ok = sum(1 for c in ids if rag[c]["refutes"] and ok(rag[c]))
    print(f"触发 case 中最终答对: 基线={trig_base_ok}/{n_base_trig}  RAG={trig_rag_ok}/{n_rag_trig}")
    print()

    # case-level delta
    print("=== case 级变化（RAG 相对基线）===")
    improved = [c for c in ids if not ok(base[c]) and ok(rag[c])]
    regressed = [c for c in ids if ok(base[c]) and not ok(rag[c])]
    print(f"RAG 修正（基线错→RAG对）: {len(improved)}  {sorted(int(c.rsplit('_',1)[-1]) for c in improved)}")
    for c in improved:
        print(f"   + id{int(c.rsplit('_',1)[-1])}: {base[c]['truth'][:50]}")
    print(f"RAG 破坏（基线对→RAG错）: {len(regressed)}  {sorted(int(c.rsplit('_',1)[-1]) for c in regressed)}")
    for c in regressed:
        print(f"   - id{int(c.rsplit('_',1)[-1])}: {base[c]['truth'][:50]}")
    print()

    # refutation alternative quality on fired cases
    print("=== refutation 触发 case 的 alternative 对比（基线 vs RAG）===")
    for c in ids:
        b_alt = base[c]["refutes"][0].get("alternative", "") if base[c]["refutes"] else ""
        r_alt = rag[c]["refutes"][0].get("alternative", "") if rag[c]["refutes"] else ""
        if b_alt or r_alt:
            print(f"id{int(c.rsplit('_',1)[-1]):2d} truth={base[c]['truth'][:35]:35s}")
            print(f"     基线alt: {b_alt[:65]}")
            print(f"     RAG alt: {r_alt[:65]}")


if __name__ == "__main__":
    main()

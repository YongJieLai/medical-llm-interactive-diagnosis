"""对 cover vs cover_debias 的错题做根因分类（对齐 error-distribution-and-improvement.md 口径）。

六类：off_system / wrong_entity / overdiagnosis / underdiagnosis / give_up / judge_false_negative。
输入 summary + truth + pred，DeepSeek 逐例标注。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_client import get_llm, chat as _llm_chat  # noqa: E402

REC = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"

MODES = ["refine_cv_cover", "refine_cv_cover_debias"]

SYSP = (
    "You are a medical error analyst. Classify the diagnostic ERROR into exactly one of "
    "six types by comparing the true diagnosis (truth) with the model's prediction (pred), "
    "using the clinical summary as context.\n"
    "Types:\n"
    "- off_system: pred is a disease of a completely different organ system / direction from truth.\n"
    "- wrong_entity: pred is in the SAME system but names the wrong disease entity.\n"
    "- overdiagnosis: pred upgrades a common/benign/functional truth into a severe/rare/organic disease.\n"
    "- underdiagnosis: pred downgrades a specific truth into a vague/broader/lower-acuity label.\n"
    "- give_up: pred gives up (e.g. 'insufficient information').\n"
    "- judge_false_negative: pred is substantively the SAME as truth (just worded differently).\n"
    'Output STRICT JSON only: {"error_type": "<one of the six>"}'
)


def load_rec(mode):
    d = {}
    for f in (REC / mode / "qwen-plus" / "16_turns" / "record").glob("*.json"):
        r = json.loads(f.read_text())
        d[r["case_id"]] = r
    return d


def summary_of(rec):
    for e in rec.get("trace", []):
        if e.get("role_id") == "summarizer":
            return e.get("content", "")
    return ""


def load_pc(mode):
    return json.loads((OUT / f"percase_{mode}.json").read_text())


def main():
    client, model = get_llm("deepseek")
    recs = {m: load_rec(m) for m in MODES}
    pcs = {m: load_pc(m) for m in MODES}

    for m in MODES:
        wrong = [c for c in pcs[m] if pcs[m][c]["correct"] is False]
        dist = {}
        for c in wrong:
            rec = recs[m][c]
            truth = rec["ground_truth"]
            pred = rec["prediction"]
            summ = summary_of(rec)
            up = (
                f"### Summary ###\n{summ}\n\n"
                f"### Truth ###\n{truth}\n\n"
                f"### Pred ###\n{pred}"
            )
            try:
                r = _llm_chat(client, model, [
                    {"role": "system", "content": SYSP},
                    {"role": "user", "content": up},
                ], temperature=0.0, max_tokens=64, json_mode=True)
                et = (r or {}).get("error_type", "unknown")
            except Exception:
                et = "unknown"
            dist[et] = dist.get(et, 0) + 1
        total = len(wrong)
        print(f"\n=== {m} 错题 {total} 例 错误类型分布 ===")
        for k in ["off_system", "wrong_entity", "overdiagnosis", "underdiagnosis", "give_up", "judge_false_negative", "unknown"]:
            n = dist.get(k, 0)
            if n:
                print(f"  {k:22s} {n:3d} ({n/total:.1%})")
        # 写缓存
        (OUT / f"errdist_{m}.json").write_text(json.dumps(dist, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

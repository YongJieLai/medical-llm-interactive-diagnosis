"""深挖 cover 底座的 wrong_entity：区分「真·错实体」 vs 「实体粒度不同（判分偏严）」。

两段式：先定 error_type（六类，对齐既有口径），对 wrong_entity 再判断 granularity 子类。
最终输出：
  1) 错误类型分布
  2) wrong_entity 子类分布（true_wrong vs granularity_mismatch）
  3) 判分偏严总量 = judge_false_negative + granularity_mismatch，及「修正后真实准确率」
每例明细 + 理由落盘 results/wrong_entity_analysis_cover.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_client import get_llm, chat as _llm_chat  # noqa: E402

REC = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"

MODE = "refine_cv_cover"

SYSP = (
    "You are a medical error analyst. Compare the true diagnosis (truth) with the model's "
    "prediction (pred), using the clinical summary as context.\n\n"
    "Step 1 — classify the error into exactly one of six types:\n"
    "- off_system: pred is a disease of a completely different organ system / direction from truth.\n"
    "- wrong_entity: pred is in the SAME organ system but names the wrong disease entity.\n"
    "- overdiagnosis: pred upgrades a common/benign/functional truth into a severe/rare/organic disease.\n"
    "- underdiagnosis: pred downgrades a specific truth into a vague/broader/lower-acuity label.\n"
    "- give_up: pred gives up (e.g. 'insufficient information').\n"
    "- judge_false_negative: pred is substantively the SAME disease as truth, just worded differently.\n\n"
    "Step 2 — ONLY if error_type is wrong_entity, decide the granularity:\n"
    "- granularity_mismatch: pred and truth are on the SAME clinical disease line, differing only in "
    "specificity / causative pathogen / anatomical sub-location / acuity staging. The pred is "
    "clinically correct 'at a coarser or finer grain' (e.g. truth='Pseudomonas keratitis', "
    "pred='microbial keratitis'; truth='community-acquired pneumonia', pred='pneumonia').\n"
    "- true_wrong: pred is a genuinely DIFFERENT disease in the same system "
    "(e.g. truth='pneumonia', pred='bronchitis'; truth='appendicitis', pred='diverticulitis').\n"
    "For non-wrong_entity types set granularity to 'n/a'.\n\n"
    'Output STRICT JSON only: {"error_type": "<...>", "granularity": "<granularity_mismatch|true_wrong|n/a>", "reason": "<one short sentence>"}'
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
    rec = load_rec(MODE)
    pc = load_pc(MODE)

    wrong_ids = sorted(c for c in pc if pc[c]["correct"] is False)
    details = {}
    dist = {}
    gran_dist = {}

    for c in wrong_ids:
        r = rec[c]
        truth = r["ground_truth"]
        pred = r["prediction"]
        summ = summary_of(r)
        up = (
            f"### Summary ###\n{summ}\n\n"
            f"### Truth ###\n{truth}\n\n"
            f"### Pred ###\n{pred}"
        )
        try:
            res = _llm_chat(client, model, [
                {"role": "system", "content": SYSP},
                {"role": "user", "content": up},
            ], temperature=0.0, max_tokens=160, json_mode=True)
            if not isinstance(res, dict):
                res = {}
            et = res.get("error_type", "unknown")
            gr = res.get("granularity", "n/a")
            rs = res.get("reason", "")
        except Exception as e:
            et, gr, rs = "unknown", "n/a", f"ERR: {e}"

        dist[et] = dist.get(et, 0) + 1
        if et == "wrong_entity":
            gran_dist[gr] = gran_dist.get(gr, 0) + 1
        details[c] = {
            "truth": truth,
            "pred": pred,
            "error_type": et,
            "granularity": gr,
            "reason": rs,
        }

    n = len(wrong_ids)
    print(f"=== {MODE} 错题 {n} 例 错误类型分布 ===")
    for k in ["off_system", "wrong_entity", "overdiagnosis", "underdiagnosis", "give_up", "judge_false_negative", "unknown"]:
        cnt = dist.get(k, 0)
        if cnt:
            print(f"  {k:22s} {cnt:3d} ({cnt/n:.1%})")

    print(f"\n=== wrong_entity 子类分布（{dist.get('wrong_entity',0)} 例）===")
    for k in ["true_wrong", "granularity_mismatch", "n/a", "unknown"]:
        cnt = gran_dist.get(k, 0)
        if cnt:
            print(f"  {k:22s} {cnt:3d}")

    # 判分偏严总量 = judge_false_negative + granularity_mismatch
    strict = dist.get("judge_false_negative", 0) + gran_dist.get("granularity_mismatch", 0)
    total = len(pc)
    correct = sum(1 for c in pc if pc[c]["correct"] is True)
    print(f"\n=== 判分口径评估 ===")
    print(f"  当前: 正确 {correct}/{total} = {correct/total:.1%}  错 {n}/{total}")
    print(f"  判分偏严 (judge_false_negative + granularity_mismatch): {strict} 例")
    adj_correct = correct + strict
    print(f"  修正后真实准确率: {adj_correct}/{total} = {adj_correct/total:.1%}  (+{strict/total:.1%})")

    (REPO / "results" / "wrong_entity_analysis_cover.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2)
    )
    print(f"\n明细已写 results/wrong_entity_analysis_cover.json")


if __name__ == "__main__":
    main()

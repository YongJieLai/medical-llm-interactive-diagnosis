"""面板补全效果对比：refine_cv_cover（原始稀疏面板）vs refine_cv_cover_panel（补全面板）。

三个维度回答「补全检查面板」是否消除环境瓶颈：
  1. 准确率（deterministic_judge）
  2. 关键检查覆盖率（key_facts.json 的 key_exam，复用 analyze_key_facts 口径）
  3. 检查命中率（医生 TEST 请求 -> reporter 返回结果的比例，补全前 72.6% 落空）

用法：python3 src/compare_panel_effect.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))

from deterministic_judge import judge  # noqa: E402
from llm_client import get_llm  # noqa: E402
from analyze_key_facts import load_cases, key_fact_ids, collected_key_ids  # noqa: E402

BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
KEYF = REPO / "results" / "coverage_analysis" / "key_facts.json"
MODES = ["refine_cv_cover", "refine_cv_cover_panel"]

_UNAVAIL = re.compile(
    r"not specified|not available|did not provide|unavailable|no result|not returned",
    re.IGNORECASE,
)


def load_records(mode):
    rec_dir = BASE / mode / "qwen-plus" / "16_turns" / "record"
    recs = {}
    if not rec_dir.exists():
        return recs
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            r = json.loads(fn.read_text())
        except Exception:
            continue
        recs[r["case_id"]] = r
    return recs


def test_hit_rate(rec):
    """返回 (hit, miss)，统计医生 TEST 请求命中/落空次数。"""
    hit = miss = 0
    trace = rec.get("trace", [])
    for i, e in enumerate(trace):
        if e.get("role_id") != "doctor":
            continue
        c = e.get("content", "")
        if "[TEST]" not in c:
            continue
        replied = False
        for j in range(i + 1, len(trace)):
            n = trace[j]
            if n.get("role_id") == "doctor":
                break
            if n.get("role_id") in ("reporter", "measurement"):
                rc = n.get("content", "")
                if _UNAVAIL.search(rc):
                    miss += 1
                elif re.search(r"\d+\.", rc):
                    hit += 1
                else:
                    miss += 1
                replied = True
                break
        if not replied:
            miss += 1
    return hit, miss


def main():
    keyf = json.loads(KEYF.read_text())
    cases = {c["case_id"]: c for c in load_cases()}

    client, model = None, None
    try:
        client, model = get_llm("deepseek")
    except Exception as e:
        print(f"[warn] DeepSeek 不可用: {e}", file=sys.stderr)

    results = {}
    for mode in MODES:
        recs = load_records(mode)
        n = len(recs)
        correct = 0
        tot_hit = tot_miss = 0
        # 关键检查覆盖率
        key_collected = 0
        key_total = 0
        for cid, rec in recs.items():
            pred = rec.get("prediction", "")
            truth = rec.get("ground_truth", "")
            res = judge(pred, truth, client=client, model=model)
            if res["correct"] is True:
                correct += 1
            h, m = test_hit_rate(rec)
            tot_hit += h
            tot_miss += m
            case = cases.get(cid)
            if case:
                kp, ke = key_fact_ids(case, keyf)
                if ke:
                    cp, ce = collected_key_ids(rec["trace"], kp, ke)
                    key_collected += len(ce)
                    key_total += len(ke)

        acc = correct / n if n else 0.0
        hit_rate = tot_hit / (tot_hit + tot_miss) if (tot_hit + tot_miss) else 0.0
        key_cov = key_collected / key_total if key_total else 0.0
        results[mode] = {
            "n": n, "correct": correct, "accuracy": acc,
            "test_hit": tot_hit, "test_miss": tot_miss, "test_hit_rate": hit_rate,
            "key_collected": key_collected, "key_total": key_total,
            "key_exam_cov": key_cov,
        }

    # 打印对比
    print("=" * 80)
    print("面板补全效果对比：refine_cv_cover(稀疏) vs refine_cv_cover_panel(补全)")
    print("=" * 80)
    hdr = f"{'指标':<16} {'稀疏面板':>14} {'补全面板':>14} {'变化':>10}"
    print(hdr)
    print("-" * 56)
    for mode, label in [("refine_cv_cover", "稀疏面板"), ("refine_cv_cover_panel", "补全面板")]:
        r = results[mode]
        print(f"{mode}: n={r['n']} acc={r['accuracy']:.1%} "
              f"命中={r['test_hit_rate']:.1%} 关键检查cov={r['key_exam_cov']:.1%}")
    print("-" * 56)
    r0, r1 = results["refine_cv_cover"], results["refine_cv_cover_panel"]
    print(f"{'准确率':<16} {r0['accuracy']:>14.1%} {r1['accuracy']:>14.1%} {r1['accuracy']-r0['accuracy']:>+9.1%}pp")
    print(f"{'检查命中率':<14} {r0['test_hit_rate']:>14.1%} {r1['test_hit_rate']:>14.1%} {r1['test_hit_rate']-r0['test_hit_rate']:>+9.1%}pp")
    print(f"{'关键检查覆盖率':<12} {r0['key_exam_cov']:>14.1%} {r1['key_exam_cov']:>14.1%} {r1['key_exam_cov']-r0['key_exam_cov']:>+9.1%}pp")

    out = REPO / "results" / "coverage_analysis" / "panel_effect_compare.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n已存: {out}")


if __name__ == "__main__":
    main()

"""Failure-case 信息采集覆盖率分析。

对每个模式 × 每个 case：
  1. 判分（deterministic_judge，规则层 + DeepSeek 兜底），并缓存 verdict
  2. 用上游 eid.metrics.coverage.compute_coverage 实时计算信息采集覆盖率
     - patient_coverage：病史事实覆盖率（patient [REFERENCE] 引用的事实编号）
     - exam_coverage：检查事实覆盖率（reporter 回复中的事实编号）
     - information_coverage_rate：综合覆盖率
  3. 分组汇总：正确 vs 错误的覆盖率对比 + 错题缺失事实明细

论文口径：AgentClinic 论文 coverage = 问诊提取到的相关信息 / 病例全部相关信息，
论文报告基线平均 67%，正确 72% vs 错误 63%。本脚本对齐该口径。

用法：
  python3 src/analyze_coverage.py [modes...]      # 指定模式（默认核心4模式）
  python3 src/analyze_coverage.py --all           # 全部 12 模式
  python3 src/analyze_coverage.py --report        # 只读缓存出报告

输出：
  results/coverage_analysis/percase_<mode>.json   # per-case 明细（verdict + coverage + missing）
  results/coverage_analysis/verdicts.json          # 判分缓存
"""
import os
import sys
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))

from deterministic_judge import judge  # noqa: E402
from llm_client import get_llm  # noqa: E402
from eid.metrics.coverage import compute_coverage  # noqa: E402

DATA = REPO / "benchmarks" / "eid-benchmark" / "data" / "agentclinic_medqa_segmented.jsonl"
BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"
OUT.mkdir(parents=True, exist_ok=True)

PAPER_MODES = ["cot", "roleplay", "react", "sc", "refine"]
REFINE_CV_MODES = [
    "refine_cv",
    "refine_cv_qwen",
    "refine_cv_no_decisive",
    "refine_cv_qwen_no_decisive",
    "refine_cv_qwen_check",
    "refine_cv_qwen_gap",
    "refine_cv_qwen_gap_refute",
    "refine_cv_cover",
]
ALL_MODES = PAPER_MODES + REFINE_CV_MODES + ["medclarify", "mediq_abstention", "uot",
                                               "medclarify_minq", "mediq_minq"]


def load_facts() -> dict[str, tuple[list[str], list[str]]]:
    """case_id -> (patient_facts, exam_facts)"""
    facts = {}
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = str(d.get("id"))
            af = d.get("atomic_facts", {}) or {}
            facts[cid] = (af.get("patient_facts", []) or [], af.get("exam_facts", []) or [])
    return facts


def load_records(mode: str, turns: int = 16) -> dict[str, dict]:
    rec_dir = BASE / mode / "qwen-plus" / f"{turns}_turns" / "record"
    out = {}
    if not rec_dir.exists():
        return out
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            d = json.loads(fn.read_text())
        except Exception:
            continue
        out[d.get("case_id", fn.stem)] = d
    return out


def load_verdict_cache() -> dict:
    p = OUT / "verdicts.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_verdict_cache(cache: dict):
    (OUT / "verdicts.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def missing_facts(collected: list[str], all_facts: list[str]) -> list[str]:
    """未采集到的事实（保留编号前缀）。"""
    col_ids = {m.group(1) for f in collected if (m := re.match(r"\s*(\d+)\.", f))}
    out = []
    for f in all_facts:
        m = re.match(r"\s*(\d+)\.", f)
        if m and m.group(1) not in col_ids:
            out.append(f)
    return out


def analyze_mode(mode: str, facts: dict, verdict_cache: dict, client, model, turns=16):
    records = load_records(mode, turns)
    percase = {}
    for cid, rec in sorted(records.items()):
        pred = rec.get("prediction", "")
        truth = rec.get("ground_truth", "")
        trace = rec.get("trace", [])

        # 判分（缓存）
        key = f"{mode}::{cid}"
        if key in verdict_cache and verdict_cache[key] is not None:
            correct = verdict_cache[key]
        else:
            res = judge(pred, truth, client=client, model=model)
            correct = res["correct"]
            verdict_cache[key] = correct
            save_verdict_cache(verdict_cache)

        # 覆盖率
        pf, ef = facts.get(cid, ([], []))
        cov = compute_coverage(trace, pf, ef)

        percase[cid] = {
            "mode": mode,
            "case_id": cid,
            "truth": truth,
            "pred": pred,
            "correct": correct,
            "patient_coverage": cov.patient_coverage,
            "exam_coverage": cov.exam_coverage,
            "information_coverage_rate": cov.information_coverage_rate,
            "n_patient_total": len(pf),
            "n_exam_total": len(ef),
            "n_patient_collected": len(cov.collected_patient_facts),
            "n_exam_collected": len(cov.collected_exam_facts),
            "missing_patient": missing_facts(cov.collected_patient_facts, pf),
            "missing_exam": missing_facts(cov.collected_exam_facts, ef),
        }

    (OUT / f"percase_{mode}.json").write_text(json.dumps(percase, ensure_ascii=False, indent=2))
    return percase


def report_mode(mode: str, percase: dict):
    corrects = [v for v in percase.values() if v["correct"] is True]
    incorrects = [v for v in percase.values() if v["correct"] is False]
    uncertain = [v for v in percase.values() if v["correct"] is None]

    def avg(lst, k):
        return sum(v[k] for v in lst) / len(lst) if lst else float("nan")

    print(f"\n{'='*92}")
    print(f"模式: {mode}   (correct={len(corrects)}  incorrect={len(incorrects)}  uncertain={len(uncertain)})")
    print(f"{'='*92}")
    print(f"{'组别':<6} {'病例数':>5} {'病史覆盖':>9} {'检查覆盖':>9} {'综合覆盖':>9}")
    print("-" * 44)
    print(f"{'正确':<6} {len(corrects):>5} {avg(corrects,'patient_coverage'):>9.1%} {avg(corrects,'exam_coverage'):>9.1%} {avg(corrects,'information_coverage_rate'):>9.1%}")
    print(f"{'错误':<6} {len(incorrects):>5} {avg(incorrects,'patient_coverage'):>9.1%} {avg(incorrects,'exam_coverage'):>9.1%} {avg(incorrects,'information_coverage_rate'):>9.1%}")
    if uncertain:
        print(f"{'不确定':<6} {len(uncertain):>5} {avg(uncertain,'patient_coverage'):>9.1%} {avg(uncertain,'exam_coverage'):>9.1%} {avg(uncertain,'information_coverage_rate'):>9.1%}")
    print("-" * 44)
    if corrects and incorrects:
        dp = avg(corrects, 'patient_coverage') - avg(incorrects, 'patient_coverage')
        de = avg(corrects, 'exam_coverage') - avg(incorrects, 'exam_coverage')
        di = avg(corrects, 'information_coverage_rate') - avg(incorrects, 'information_coverage_rate')
        print(f"正确-错误差值: 病史 {dp:+.1%}  检查 {de:+.1%}  综合 {di:+.1%}")

    # 错题明细（按综合覆盖率升序）
    print(f"\n--- 错题明细（按综合覆盖率升序，前 30）---")
    inc_sorted = sorted(incorrects, key=lambda v: v["information_coverage_rate"])
    for v in inc_sorted[:30]:
        print(f"\n[{v['case_id']}] 综合覆盖={v['information_coverage_rate']:.1%} "
              f"病史={v['patient_coverage']:.1%}({v['n_patient_collected']}/{v['n_patient_total']}) "
              f"检查={v['exam_coverage']:.1%}({v['n_exam_collected']}/{v['n_exam_total']})")
        print(f"  truth: {v['truth'][:70]!r}")
        print(f"  pred : {v['pred'][:70]!r}")
        if v["missing_patient"]:
            print(f"  缺失病史({len(v['missing_patient'])}): " + " | ".join(v["missing_patient"][:8]))
        if v["missing_exam"]:
            print(f"  缺失检查({len(v['missing_exam'])}): " + " | ".join(v["missing_exam"][:8]))

    return percase


def main():
    args = sys.argv[1:]
    if not args:
        modes = ["refine", "refine_cv_qwen_no_decisive", "sc", "roleplay"]
    elif args[0] == "--all":
        modes = ALL_MODES
    elif args[0] == "--report":
        for p in sorted(OUT.glob("percase_*.json")):
            m = p.stem.replace("percase_", "")
            percase = json.loads(p.read_text())
            report_mode(m, percase)
        return
    else:
        modes = args

    facts = load_facts()
    verdict_cache = load_verdict_cache()

    client, model = None, None
    try:
        client, model = get_llm("deepseek")
    except Exception as e:
        print(f"[warn] DeepSeek 不可用，规则判不出的 case 记为 uncertain: {e}", file=sys.stderr)

    for m in modes:
        percase = analyze_mode(m, facts, verdict_cache, client, model)
        report_mode(m, percase)

    save_verdict_cache(verdict_cache)
    print(f"\nper-case 明细已存: {OUT}")


if __name__ == "__main__":
    main()

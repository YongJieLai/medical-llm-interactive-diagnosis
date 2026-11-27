"""对论文 5 种模式的跑批结果做 deterministic_judge 判分并汇总。

口径：统一走 deterministic_judge.judge（规则层 + DeepSeek 兜底双判），
与历史实验一致（is_correct 字段不可信）。

特殊处理：cot（Upper Bound）的 prediction 是 THOUGHT+DIAGNOSIS 全文（上游 extract_action
不匹配 "DIAGNOSIS:" 无方括号格式而 fallback 全文），判分前先提取 "DIAGNOSIS:" 之后的
最终诊断，保证与其它模式（精确诊断）同口径。

用法：
  python3 src/summarize_paper_modes.py [base_dir]
输出：各模式 准确率 / 判定来源分布 / 不确定(需 LLM)case 列表。
"""
import os
import sys
import re
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from deterministic_judge import judge, rule_judge, normalize  # noqa: E402
from llm_client import get_llm  # noqa: E402

PAPER_MODES = ["cot", "roleplay", "react", "sc", "refine"]

# 我们改过的方法（refine_cv 系列），在 16/200 下重标定。
REFINE_CV_MODES = [
    "refine_cv",
    "refine_cv_qwen",
    "refine_cv_no_decisive",
    "refine_cv_qwen_no_decisive",
    "refine_cv_qwen_check",
    "refine_cv_qwen_gap",
    "refine_cv_qwen_gap_refute",
    "react_cv",
    "roleplay_cv",
    "refine_cv_staged",
    "refine_cv_staged_k4",
    "refine_cv_staged_k8",
    "refine_cv_cover",
    "refine_cv_freebudget",
]

# 外部论文方法（零训练移植 baseline）
EXTERNAL_MODES = ["medclarify", "mediq_abstention", "uot",
                  "medclarify_minq", "mediq_minq"]

ALL_MODES = PAPER_MODES + REFINE_CV_MODES + EXTERNAL_MODES
BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"


def extract_cot_diagnosis(pred: str) -> str:
    """cot 全文里提取 'DIAGNOSIS: ...' 后的最终诊断；提取失败则返回原文。"""
    m = re.search(r"DIAGNOSIS:\s*(.+)", pred, re.DOTALL)
    if m:
        return m.group(1).strip()
    return pred


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


def main():
    turns = 16
    # 可选参数：paper(论文5模式) / cv(refine_cv系列) / all(默认全部)
    subset = sys.argv[1] if len(sys.argv) > 1 else "all"
    if subset == "paper":
        modes = PAPER_MODES
    elif subset == "cv":
        modes = REFINE_CV_MODES
    elif subset == "ext":
        modes = EXTERNAL_MODES
    else:
        modes = ALL_MODES

    deepseek_client, deepseek_model = None, None
    try:
        deepseek_client, deepseek_model = get_llm("deepseek")
    except Exception as e:
        print(f"[warn] DeepSeek 不可用，规则判不出的 case 记为 uncertain: {e}", file=sys.stderr)

    summary = {}
    for mode in modes:
        records = load_records(mode, turns)
        total = len(records)
        correct = 0
        incorrect = 0
        uncertain = 0
        src_counter = {}
        uncertain_cases = []
        err_cases = []

        for cid, rec in records.items():
            pred = rec.get("prediction", "")
            truth = rec.get("ground_truth", "")
            if "error" in rec and not pred:
                err_cases.append(cid)
                continue
            if mode == "cot":
                pred = extract_cot_diagnosis(pred)

            res = judge(pred, truth, client=deepseek_client, model=deepseek_model)
            v = res["correct"]
            src = res["source"]
            if v is True:
                correct += 1
            elif v is False:
                incorrect += 1
            else:
                uncertain += 1
                uncertain_cases.append((cid, pred, truth))
            src_counter[src] = src_counter.get(src, 0) + 1

        acc = correct / total if total else 0.0
        summary[mode] = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "uncertain": uncertain,
            "err": len(err_cases),
            "accuracy": acc,
            "source_dist": src_counter,
        }
        print(f"\n=== {mode} ===")
        print(f"  total={total}  correct={correct}  incorrect={incorrect}  uncertain={uncertain}  err={len(err_cases)}")
        print(f"  accuracy = {correct}/{total} = {acc:.1%}")
        print(f"  判定来源分布: {src_counter}")
        if uncertain_cases:
            print(f"  规则判不出(uncertain) case: {len(uncertain_cases)}")
            for cid, p, t in uncertain_cases[:20]:
                print(f"    - {cid}: pred={p[:60]!r} truth={t[:60]!r}")
        if err_cases:
            print(f"  error case: {err_cases[:10]}")

    # 汇总对比表
    print("\n\n=== 汇总对比（deterministic_judge，200 例 / 16 turns，qwen-plus 同模型）===")
    print(f"{'mode':10s} {'论文方法':18s} {'acc':>8s} {'correct':>8s} {'total':>6s}")
    name_map = {
        "cot": "Upper Bound(静态)",
        "roleplay": "Baseline(交互)",
        "react": "ReAct",
        "sc": "SC(总结诊断)",
        "refine": "REFINE(论文)",
        "refine_cv": "CV 跨模型四检查",
        "refine_cv_qwen": "CV 同模型四检查",
        "refine_cv_no_decisive": "CV 跨模型三检查",
        "refine_cv_qwen_no_decisive": "CV 同模型三检查",
        "refine_cv_qwen_check": "CV 只claim",
        "refine_cv_qwen_gap": "CV 只gap",
        "refine_cv_qwen_gap_refute": "CV gap+refute",
        "react_cv": "ReAct + CEV-GR四检查",
        "roleplay_cv": "Roleplay + CEV-GR四检查",
        "refine_cv_staged": "两阶段(K=6)",
        "refine_cv_staged_k4": "两阶段(K=4)",
        "refine_cv_staged_k8": "两阶段(K=8)",
        "refine_cv_cover": "广覆盖采集(方案A)",
        "refine_cv_freebudget": "去效率无框架(消融)",
        "medclarify": "MedClarify(DEIG)",
        "mediq_abstention": "MEDIQ(abstention)",
        "uot": "UoT(信息增益)",
        "medclarify_minq": "MedClarify(强制问4轮)",
        "mediq_minq": "MEDIQ(强制问4轮)",
    }
    for mode in modes:
        s = summary[mode]
        print(f"{mode:10s} {name_map[mode]:18s} {s['accuracy']:8.1%} {s['correct']:8d} {s['total']:6d}")

    # 存汇总 json（按 subset 命名，避免互相覆盖）
    out = REPO / "results" / f"paper_modes_judge_summary_{subset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n汇总已存: {out}")


if __name__ == "__main__":
    main()

"""用确定性 judge（规则 + DeepSeek 兜底）重判 refine_cv_qwen 全部 60 例。

背景：record 里的 is_correct 字段是 AgentClinic 原生 judge，把大量"实质正确"
的 case 误判为 False（实测 21 个误杀）。这污染了之前"从头错"分析所用的错误
case 集。本脚本重建可靠的错误 case 集，作为"从头错"复核的地基。

输出：results/rejudge_qwen.json —— 60 例的 {case, pred, truth, correct, source}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek

BASE = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "eid-benchmark" / "results" / "medqa"
    / "refine_cv_qwen" / "qwen-plus" / "10_turns" / "record"
)


def main():
    client, model = get_deepseek()
    files = sorted(BASE.glob("*.json"))
    print(f"重判 refine_cv_qwen 共 {len(files)} 例（规则 + DeepSeek 兜底）\n")

    detail = []
    correct = incorrect = uncertain = 0
    for fn in files:
        r = json.loads(fn.read_text())
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client=client, model=model)
        v = res["correct"]
        if v is True:
            correct += 1
        elif v is False:
            incorrect += 1
        else:
            uncertain += 1
        detail.append({
            "case": r.get("case_id"),
            "pred": pred,
            "truth": truth,
            "correct": v,
            "source": res["source"],
            "native_is_correct": r.get("is_correct"),
        })

    print(f"可靠判定结果：")
    print(f"  CORRECT:   {correct}")
    print(f"  INCORRECT: {incorrect}")
    print(f"  UNCERTAIN: {uncertain}")
    print()

    wrong = [d for d in detail if d["correct"] is False]
    print(f"=== 可靠错误 case（{len(wrong)} 个）===")
    for d in wrong:
        print(f"  {d['case']:28s} GT={d['truth']!r:45s} -> pred={d['pred']!r}")

    outp = Path(__file__).resolve().parent.parent / "results" / "rejudge_qwen.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(detail, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

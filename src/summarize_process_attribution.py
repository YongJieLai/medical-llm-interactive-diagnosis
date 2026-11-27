"""汇总过程归因结果：区分「没采到关键检查」的真实机制。

把「没采到的判别性检查」按机制拆成：
  非缺陷：redundant_closed（当时最优）/ env_unavailable（环境约束）
  真问题：turn_exhausted（轮次耗尽）/ wrong_priority（优先级错配）/ direction_not_considered（方向未想到）
"""
import json
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "coverage_analysis"
MERGED = OUT / "process_attribution.json"

MECH_ORDER = ["redundant_closed", "env_unavailable", "turn_exhausted",
              "wrong_priority", "direction_not_considered", "other"]
MECH_CN = {
    "redundant_closed": "冗余闭合(当时最优)",
    "env_unavailable": "环境不可用(非模型缺陷)",
    "turn_exhausted": "轮次耗尽(采集规划)",
    "wrong_priority": "优先级错配(决策取舍)",
    "direction_not_considered": "方向未想到(真知识缺口)",
    "other": "其他",
}
NON_DEFECT = {"redundant_closed", "env_unavailable"}


def main():
    data = json.loads(MERGED.read_text())
    n = len(data)
    print(f"过程归因条目数（判别性检查）: {n}\n")

    mech = Counter(v.get("mechanism") for v in data.values())
    print("=== 机制分布 ===")
    for m in MECH_ORDER:
        c = mech.get(m, 0)
        if c:
            print(f"  {m:24s} {c:4d}  {c/n:.1%}  ({MECH_CN[m]})")

    # 非缺陷 vs 真问题
    non_def = sum(mech.get(m, 0) for m in NON_DEFECT)
    real = n - non_def
    print(f"\n=== 核心结论 ===")
    print(f"  非缺陷（当时最优 + 环境约束）: {non_def}  ({non_def/n:.1%})")
    print(f"  真问题（轮次/优先级/方向没想到）: {real}  ({real/n:.1%})")
    print(f"    其中方向未想到(真知识缺口): {mech.get('direction_not_considered',0)}  ({mech.get('direction_not_considered',0)/n:.1%})")

    # 按 case 正确性交叉
    print("\n=== 按所在 case 是否答对 交叉 ===")
    for ok in (True, False):
        sub = [v for v in data.values() if v.get("correct") is ok]
        if not sub:
            continue
        sm = Counter(v.get("mechanism") for v in sub)
        label = "正确case" if ok else "错误case"
        print(f"  {label} (n={len(sub)}):")
        for m in MECH_ORDER:
            c = sm.get(m, 0)
            if c:
                print(f"      {m:24s} {c:4d}  {c/len(sub):.1%}")

    # 输出到文件
    (OUT / "process_attribution_summary.json").write_text(
        json.dumps({"n": n, "mechanism": dict(mech)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

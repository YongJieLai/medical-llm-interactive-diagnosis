"""合并分片结果 + 汇总统计「未采到的关键检查」定性分类。

读 missed_key_exams_classified_{0..3}.json → 合并 → 统计 role/retrievability，
回答 B（知识缺口）vs C（信息上界）。
"""
import json
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "coverage_analysis"
SHARDS = [OUT / f"missed_key_exams_classified_{i}.json" for i in range(4)]
MERGED = OUT / "missed_key_exams_classified.json"


def merge():
    merged = {}
    for p in SHARDS:
        if p.exists():
            merged.update(json.loads(p.read_text()))
    MERGED.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return merged


def main():
    merged = merge()
    print(f"已分类条目数: {len(merged)}")

    role = Counter(v.get("role") for v in merged.values())
    print("\n=== role 分布（未采到的关键检查里，各角色占比）===")
    for k, n in role.most_common():
        print(f"  {k:14s} {n:4d}  {n/len(merged):.1%}")

    # confirmatory 里的 retrievability
    conf = [v for v in merged.values() if v.get("role") == "confirmatory"]
    print(f"\n=== confirmatory（判别性检查）共 {len(conf)} 条，retrievability 分布 ===")
    retr = Counter(v.get("retrievability") for v in conf)
    for k, n in retr.most_common():
        print(f"  {k:14s} {n:4d}  {n/len(conf):.1%}" if conf else "  无")

    # 按 case 正确性交叉
    print("\n=== 按所在 case 是否答对 交叉 ===")
    for ok in (True, False):
        sub = [v for v in merged.values() if v.get("correct") is ok]
        if not sub:
            continue
        r = Counter(v.get("role") for v in sub)
        c = [v for v in sub if v.get("role") == "confirmatory"]
        rr = Counter(v.get("retrievability") for v in c)
        label = "正确case" if ok else "错误case"
        print(f"  {label} (n={len(sub)}): role={dict(r)} | confirmatory里 {dict(rr)}")

    # 结论性统计：真正判别性检查未采到 vs 常规/阴性误标
    n_routine = role.get("routine", 0) + role.get("negative", 0)
    n_unc = role.get("uncertain", 0)
    n_conf = role.get("confirmatory", 0)
    print("\n=== 结论 ===")
    print(f"  未采到总数: {len(merged)}")
    print(f"  常规基线/阴性排除（key_facts 标注噪音）: {n_routine}  ({n_routine/len(merged):.1%})")
    print(f"  不确定: {n_unc}")
    print(f"  真正判别性检查未采到: {n_conf}  ({n_conf/len(merged):.1%})")
    if conf:
        n_std = retr.get("standard", 0)
        n_spe = retr.get("specialized", 0)
        print(f"    其中 standard(常规该想到)={n_std}, specialized(罕见想不到)={n_spe}")


if __name__ == "__main__":
    main()

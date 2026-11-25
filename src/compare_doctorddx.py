"""对比 doctorddx（DDx 驱动问诊）vs baseline（gap_refute），同口径确定性 judge。

变体（case_id 对齐，均为 qwen-plus / 10_turns / 前 30 例）：
  - refine_cv_qwen_gap_refute : baseline（无证据采集修正）
  - refine_cv_qwen_gap_refute_doctorddx : +DDx 驱动问诊（P3，治证据采集不足 46%）

判读：doctorddx 净贡献是否为正；重点看「证据采集不足」的 12 个 case
（medqa_20/21/28/32/37/40/41/50/57/58/7/9）是否被救回。

运行：python3 src/compare_doctorddx.py [N=30]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import judge, get_deepseek  # noqa: E402

BASE = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "eid-benchmark" / "results" / "medqa"
)

BASE_METHOD = "refine_cv_qwen_gap_refute"
DDX_METHOD = "refine_cv_qwen_gap_refute_doctorddx"

# 证据采集不足的 12 个 case（fromhead-reaudit.md）
GATHER_GAP_CASES = {"medqa_" + c for c in
    ["20", "21", "28", "32", "37", "40", "41", "50", "57", "58", "7", "9"]}


def load_record(method: str, i: int) -> dict | None:
    fn = BASE / method / "qwen-plus" / "10_turns" / "record" / f"agentclinic_medqa_{i}.json"
    if not fn.exists():
        return None
    return json.loads(fn.read_text())


def judge_method(client, model, method: str, n: int):
    stats = Counter()
    detail = {}
    for i in range(n):
        r = load_record(method, i)
        if r is None:
            stats["missing"] += 1
            continue
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        res = judge(pred, truth, client=client, model=model)
        v = res["correct"]
        stats["correct" if v is True else ("incorrect" if v is False else "uncertain")] += 1
        cid = r.get("case_id", f"medqa_{i}")
        detail[cid] = {"pred": pred, "truth": truth, "correct": v, "source": res["source"]}
    return stats, detail


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    client, model = get_deepseek()
    print(f"doctorddx vs baseline（前 {n} 例，确定性 judge + DeepSeek 兜底）\n")

    b_stats, b_detail = judge_method(client, model, BASE_METHOD, n)
    d_stats, d_detail = judge_method(client, model, DDX_METHOD, n)

    def acc(s):
        tot = s["correct"] + s["incorrect"] + s["uncertain"]
        return s["correct"] / tot * 100 if tot else 0

    b_acc, d_acc = acc(b_stats), acc(d_stats)
    print(f"[baseline {BASE_METHOD}]  correct={b_stats['correct']}  "
          f"incorrect={b_stats['incorrect']}  uncertain={b_stats['uncertain']}  acc={b_acc:.1f}%")
    print(f"[doctorddx {DDX_METHOD}]  correct={d_stats['correct']}  "
          f"incorrect={d_stats['incorrect']}  uncertain={d_stats['uncertain']}  acc={d_acc:.1f}%")
    print()
    print(f"净贡献 = {d_acc - b_acc:+.1f} pp")
    print()

    # case 级翻转
    better = worse = 0
    better_cases, worse_cases = [], []
    for cid in d_detail:
        a, b = d_detail[cid], b_detail.get(cid)
        if b is None:
            continue
        if a["correct"] != b["correct"]:
            if a["correct"] and not b["correct"]:
                better += 1
                better_cases.append(cid)
            elif b["correct"] and not a["correct"]:
                worse += 1
                worse_cases.append(cid)
    print(f"救回 {better} 个，掰歪 {worse} 个")
    print(f"救回清单: {sorted(better_cases)}")
    print(f"掰歪清单: {sorted(worse_cases)}")
    print()
    # 证据采集不足的 12 个 case 是否被救回
    gg_saved = [c for c in better_cases if c in GATHER_GAP_CASES]
    print(f"其中「证据采集不足」12 case 被救回: {len(gg_saved)} 个 -> {sorted(gg_saved)}")

    out = {"baseline": b_detail, "doctorddx": d_detail,
           "baseline_acc": b_acc, "doctorddx_acc": d_acc,
           "better": sorted(better_cases), "worse": sorted(worse_cases)}
    outp = Path(__file__).resolve().parent.parent / "results" / "doctorddx_compare.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n完整明细已写入 {outp}")


if __name__ == "__main__":
    main()

"""定性分析：方案A（refine_cv_cover）里「没采到的关键检查」清单。

回答上一轮的核心问题：38.9% 的关键检查没采到，是「常规检查被漏了」
（假设B 知识缺口/采集策略问题，可引入"症状→检查"临床路径补），还是
「罕见特异性检查想不到」（假设C 信息上界，prompt 天花板）。

第一步（本脚本，纯规则、无 LLM）：
  把每个 case 里 key_exam 标注为关键、但 trace 里没被采到的检查逐条列出，
  连同「检查内容文本 + 正确诊断 + 该例是否答对」，落盘供下一步 LLM 定性分类。

复用 analyze_key_facts 的 collected_key_ids 口径，保证与覆盖率统计一致。
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))

from analyze_key_facts import load_cases, key_fact_ids, collected_key_ids  # noqa: E402

DATA = REPO / "benchmarks" / "eid-benchmark" / "data" / "agentclinic_medqa_segmented.jsonl"
BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"
KEYF = OUT / "key_facts.json"
MODE = "refine_cv_cover"


def load_record(mode):
    rec_dir = BASE / mode / "qwen-plus" / "16_turns" / "record"
    recs = {}
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            r = json.loads(fn.read_text())
        except Exception:
            continue
        recs[r["case_id"]] = r
    return recs


def load_correct(mode):
    p = OUT / f"percase_{mode}.json"
    if not p.exists():
        return {}
    return {c: v.get("correct") for c, v in json.loads(p.read_text()).items()}


def exam_text_by_id(exam_facts, num):
    """按编号取检查内容文本（去掉编号前缀）。"""
    for f in exam_facts:
        m = re.match(r"\s*(\d+)\.\s*(.*)", f)
        if m and m.group(1) == str(num):
            return m.group(2).strip()
    return None


def main():
    keyf = json.loads(KEYF.read_text())
    cases = {c["case_id"]: c for c in load_cases()}
    recs = load_record(MODE)
    corr = load_correct(MODE)

    missed = []          # 每条没采到的 key_exam
    percase_summary = []  # 每例汇总
    tot_key = 0
    tot_missed = 0

    for cid, case in cases.items():
        kp, ke = key_fact_ids(case, keyf)
        if not ke:
            continue
        rec = recs.get(cid)
        if not rec:
            continue
        cp, ce = collected_key_ids(rec["trace"], kp, ke)
        # ce 里的编号可能是 str；key_exam 编号可能是 int
        collected = {str(x) for x in ce}
        all_key = {str(x) for x in ke}
        missed_ids = sorted(all_key - collected, key=lambda x: int(x))
        tot_key += len(all_key)
        tot_missed += len(missed_ids)

        for num in missed_ids:
            txt = exam_text_by_id(case["exam_facts"], num)
            missed.append({
                "case_id": cid,
                "truth": case["truth"],
                "correct": corr.get(cid),
                "fact_num": int(num),
                "exam_text": txt or "(未找到文本)",
            })

        percase_summary.append({
            "case_id": cid,
            "truth": case["truth"],
            "correct": corr.get(cid),
            "n_key_exam": len(all_key),
            "n_missed": len(missed_ids),
            "missed_nums": [int(x) for x in missed_ids],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "missed_key_exams_cover.json").write_text(
        json.dumps(missed, ensure_ascii=False, indent=1))
    (OUT / "missed_key_exams_cover_percase.json").write_text(
        json.dumps(percase_summary, ensure_ascii=False, indent=1))

    # 简单统计
    print(f"模式 {MODE}")
    print(f"总关键检查数: {tot_key}")
    print(f"未采到数: {tot_missed}  ({tot_missed/tot_key:.1%})")
    print(f"涉及 case 数: {sum(1 for p in percase_summary if p['n_missed']>0)}/{len(percase_summary)}")
    # 未采到检查的正确答案分布
    from collections import Counter
    c_corr = Counter("正确" if m["correct"] else "错误" for m in missed)
    print(f"未采到检查所在 case 的正确性: {dict(c_corr)}")
    print(f"\n清单已存: {OUT/'missed_key_exams_cover.json'}")
    print(f"逐例汇总已存: {OUT/'missed_key_exams_cover_percase.json'}")


if __name__ == "__main__":
    main()

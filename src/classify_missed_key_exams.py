"""第二步：对「未采到的关键检查」做 LLM 定性分类。

回答核心问题（B 知识缺口 vs C 信息上界）：
  362 条未采到的 key_exam，逐条标注：
    role:  confirmatory（判别性检查，指向确诊）/ routine（常规基线/生命体征/一般筛查）
           / negative（阴性排除性）/ uncertain
    retrievability（仅 confirmatory）:
           standard（教科书标准路径该想到） / specialized（高度特异/罕见，想不到）

据此回答两件事：
  1. 38.9% 的「关键检查未采到」里，有多少其实是 routine/negative（key_facts 标注
     把常规项误标为关键），真正「判别性检查未采到」是多少 → 修正 gap 大小。
  2. 真正没采到的判别性检查里，standard vs specialized 的比例 → B vs C。

用法（多进程分片，断点续跑，每 case 独立写回）：
  for i in 0 1 2 3; do SHARD_ID=$i SHARD_TOTAL=4 python3 src/classify_missed_key_exams.py & done; wait
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_client import get_llm, chat  # noqa: E402

DATA = REPO / "benchmarks" / "eid-benchmark" / "data" / "agentclinic_medqa_segmented.jsonl"
OUT = REPO / "results" / "coverage_analysis"
MISSED = OUT / "missed_key_exams_cover.json"
RESULT = OUT / "missed_key_exams_classified.json"

_SYS = (
    "You are a senior clinician. Given a patient's chief complaint, the correct final "
    "diagnosis, and ONE examination/test result that the doctor FAILED to collect during "
    "a diagnostic interview, classify this exam/test along two axes.\n\n"
    "role (what this result is FOR):\n"
    "- confirmatory: this result directly confirms or strongly points to the given diagnosis; "
    "it is a DISCRIMINATIVE test one orders because one suspects this specific disease.\n"
    "- routine: a routine baseline / vital sign / general screening lab that would be ordered "
    "for almost any similar presentation, NOT specific to this diagnosis (e.g. temperature, "
    "blood pressure, heart rate, CBC, electrolytes, basic metabolic panel).\n"
    "- negative: the result is normal/negative, and its value is to EXCLUDE other conditions "
    "rather than to confirm the diagnosis (e.g. \"culture negative\", \"CA-125 normal\", "
    "\"cranial nerves intact\").\n"
    "- uncertain: cannot decide.\n\n"
    "retrievability (fill ONLY if role == confirmatory):\n"
    "- standard: for this chief complaint, a textbook/standard clinical-pathway test that a "
    "competent clinician would REASONABLY think to order (even if some miss it).\n"
    "- specialized: a highly specific / rare / subspecialty test that requires deep expert "
    "knowledge or rare-disease awareness to even think of.\n"
    "- na: for non-confirmatory items.\n\n"
    'Output JSON only: {"role": "...", "retrievability": "..."}\n'
)


def load_chief() -> dict:
    """case_id -> 主诉摘要（用于判断标准路径）。"""
    out = {}
    for line in DATA.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        cid = str(d.get("id"))
        o = d.get("OSCE_Examination", {}) or {}
        pa = o.get("Patient_Actor", {}) or {}
        sym = pa.get("Symptoms", {}) or {}
        hist = (pa.get("History") or "").strip()
        prim = sym.get("Primary_Symptom", "") or ""
        secs = sym.get("Secondary_Symptoms", []) or []
        obj = (o.get("Objective_for_Doctor") or "").strip()
        parts = [obj, f"Primary: {prim}"]
        if secs:
            parts.append("Also: " + "; ".join(secs[:6]))
        if hist:
            parts.append("History: " + hist[:200])
        out[cid] = " | ".join(p for p in parts if p)
    return out


def classify_one(item, chief, client, model):
    prompt = (
        f"Chief complaint / presentation:\n{chief}\n\n"
        f"Correct diagnosis: {item['truth']}\n\n"
        f"Examination/test result the doctor FAILED to collect:\n{item['exam_text']}\n\n"
        "Classify this exam/test."
    )
    msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}]
    raw = chat(client, model, msgs, temperature=0.0, max_tokens=300, json_mode=True)
    if isinstance(raw, dict):
        return raw
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"role": "uncertain", "retrievability": "na"}


def main():
    shard_id = int(os.getenv("SHARD_ID", "0"))
    shard_total = int(os.getenv("SHARD_TOTAL", "1"))

    items = json.loads(MISSED.read_text())
    chief = load_chief()

    # 每分片写独立文件，避免并发写冲突；最后用 --merge 合并。
    shard_result = OUT / f"missed_key_exams_classified_{shard_id}.json"

    cache = {}
    if shard_result.exists():
        try:
            cache = json.loads(shard_result.read_text())
        except Exception:
            cache = {}

    my_items = items[shard_id::shard_total]
    todo = []
    for it in my_items:
        key = f"{it['case_id']}#{it['fact_num']}"
        if key not in cache:
            todo.append(it)

    print(f"[shard {shard_id}/{shard_total}] 总={len(my_items)} 待分类={len(todo)}", flush=True)

    client, model = get_llm("deepseek")
    done = 0
    for it in todo:
        key = f"{it['case_id']}#{it['fact_num']}"
        c = chief.get(it["case_id"], "")
        try:
            r = classify_one(it, c, client, model)
        except Exception as e:
            print(f"[err] {key}: {e}", file=sys.stderr, flush=True)
            r = {"role": "uncertain", "retrievability": "na"}
        cache[key] = {"truth": it["truth"], "exam_text": it["exam_text"],
                      "correct": it["correct"], "role": r.get("role", "uncertain"),
                      "retrievability": r.get("retrievability", "na")}
        done += 1
        if done % 25 == 0:
            shard_result.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            print(f"[shard {shard_id}] {done}/{len(todo)}", flush=True)

    shard_result.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    print(f"[shard {shard_id}] DONE {done}", flush=True)


if __name__ == "__main__":
    main()

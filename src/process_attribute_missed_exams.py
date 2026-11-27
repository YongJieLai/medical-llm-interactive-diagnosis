"""过程归因：为什么「没采到的关键检查」没被采到？

上一轮（analyze_missed_key_exams + classify_missed_key_exams）是「结果归因」——
站在正确答案事后判断「这个检查是标准路径该想到」，犯 hindsight bias。
它无法区分：模型「知道但主动不查」（当时最优）、「想到但轮次/环境所限」、
「优先级选错」、「根本没想到方向」。

本轮改为「过程归因」：对每条没采到的关键检查，回看该 case 的 doctor 完整
[THOUGHT] 决策轨迹，判断它当时的认知状态，分类为：

  redundant_closed   已通过其他证据闭合该诊断，明确判断此检查冗余 → 非缺陷（当时最优）
  turn_exhausted     想到该方向/检查，但排在后面，16 轮用完没做上 → 采集规划问题
  env_unavailable    请求了该检查（或等价），环境返回 unavailable → 非模型缺陷
  wrong_priority     当时有更高优先假设，正确方向被压/延后 → 决策取舍问题
  direction_not_considered  正确诊断方向从未出现在推理里 → 真知识缺口
  other              无法归类

关键：判断必须锚定在 THOUGHT 轨迹文本上（模型当时可观测的认知状态），
不许用「事后该查什么」来反推。

用法（多进程分片，每分片独立写文件，最后 --merge）：
  for i in 0 1 2 3; do SHARD_ID=$i SHARD_TOTAL=4 python3 src/process_attribute_missed_exams.py & done; wait
"""
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_client import get_llm, chat  # noqa: E402

BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
OUT = REPO / "results" / "coverage_analysis"
CLASSIFIED = OUT / "missed_key_exams_classified.json"
MODE = "refine_cv_cover"

_SYS = (
    "You are analyzing WHY a diagnostic agent failed to collect ONE specific examination/test. "
    "You are given the agent's complete reasoning trace (each turn shows its [THOUGHT] — its "
    "hypotheses, priorities, and rationale at that moment) plus the action it took.\n\n"
    "Classify the reason the agent did NOT collect the target test into exactly one mechanism:\n"
    "- redundant_closed: the agent already closed this diagnosis via other evidence and explicitly "
    "judged this test redundant / low marginal value (often the case is then answered correctly).\n"
    "- turn_exhausted: the agent considered this direction or test but ran out of turns before doing it.\n"
    "- env_unavailable: the agent requested this test (or an equivalent) but the environment returned "
    "\"unavailable / not specified / no result\".\n"
    "- wrong_priority: the agent had a higher-priority alternative hypothesis and deliberately validated "
    "that instead, leaving the correct direction deprioritized or delayed.\n"
    "- direction_not_considered: the correct diagnosis direction never appears in the agent's reasoning "
    "at all.\n"
    "- other: none of the above.\n\n"
    "BASE YOUR JUDGMENT ONLY ON THE PROVIDED TRACE. Do NOT use hindsight (\"what it should have checked\"). "
    "Quote the trace words that justify your choice.\n\n"
    'Output JSON only: {"mechanism": "...", "evidence": "<short quote from trace>"}\n'
)


def load_records():
    recs = {}
    rec_dir = BASE / MODE / "qwen-plus" / "16_turns" / "record"
    for fn in sorted(rec_dir.glob("*.json")):
        try:
            r = json.loads(fn.read_text())
        except Exception:
            continue
        recs[r["case_id"]] = r
    return recs


def extract_trace_text(rec, max_per_turn=280, max_turns=16):
    """把 doctor 轨迹压缩成文本：每轮 [ACTION] + THOUGHT 摘要。"""
    parts = []
    for e in rec.get("trace", []):
        if e.get("role_id") != "doctor":
            continue
        c = e.get("content", "")
        m = re.search(r"\[THOUGHT\](.*?)(?=\[QUERY\]|\[TEST\]|\[FINISH\]|$)", c, re.DOTALL)
        t = (m.group(1).strip() if m else c).strip()
        if len(t) > max_per_turn:
            t = t[:max_per_turn] + "…"
        if "[FINISH]" in c:
            act = "FINISH"
        elif "[TEST]" in c:
            act = "TEST"
        elif "[QUERY]" in c:
            act = "QUERY"
        else:
            act = "?"
        parts.append(f"[{act}] {t}")
        if len(parts) >= max_turns:
            break
    return "\n".join(parts)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merged = {}
        for i in range(4):
            p = OUT / f"process_attribution_{i}.json"
            if p.exists():
                merged.update(json.loads(p.read_text()))
        (OUT / "process_attribution.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=1))
        print(f"merged {len(merged)}")
        return

    shard_id = int(os.getenv("SHARD_ID", "0"))
    shard_total = int(os.getenv("SHARD_TOTAL", "1"))

    classified = json.loads(CLASSIFIED.read_text())
    # 只对 confirmatory（判别性）检查做过程归因，routine/negative 跳过
    items = []
    for k, v in classified.items():
        if v.get("role") != "confirmatory":
            continue
        cid = k.split("#")[0]
        items.append({"key": k, "cid": cid, "truth": v["truth"],
                      "exam_text": v["exam_text"], "correct": v["correct"]})

    recs = load_records()
    shard_result = OUT / f"process_attribution_{shard_id}.json"
    cache = {}
    if shard_result.exists():
        try:
            cache = json.loads(shard_result.read_text())
        except Exception:
            cache = {}

    my_items = items[shard_id::shard_total]
    todo = [it for it in my_items if it["key"] not in cache]
    print(f"[shard {shard_id}/{shard_total}] confirmatory总={len(my_items)} 待归因={len(todo)}", flush=True)

    client, model = get_llm("deepseek")
    done = 0
    for it in todo:
        rec = recs.get(it["cid"])
        trace_text = extract_trace_text(rec) if rec else "(no trace)"
        prompt = (
            f"Correct diagnosis: {it['truth']}\n\n"
            f"Target test the agent FAILED to collect:\n{it['exam_text']}\n\n"
            f"Agent's reasoning trace (chronological):\n{trace_text}\n\n"
            "Why did the agent NOT collect this test? Classify the mechanism."
        )
        msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}]
        try:
            raw = chat(client, model, msgs, temperature=0.0, max_tokens=400, json_mode=True)
        except Exception as e:
            print(f"[err] {it['key']}: {e}", file=sys.stderr, flush=True)
            raw = None
        r = raw if isinstance(raw, dict) else {"mechanism": "other", "evidence": ""}
        m = r.get("mechanism", "other")
        cache[it["key"]] = {"truth": it["truth"], "exam_text": it["exam_text"],
                            "correct": it["correct"], "mechanism": m,
                            "evidence": r.get("evidence", "")[:200]}
        done += 1
        if done % 20 == 0:
            shard_result.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            print(f"[shard {shard_id}] {done}/{len(todo)}", flush=True)

    shard_result.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    print(f"[shard {shard_id}] DONE {done}", flush=True)


if __name__ == "__main__":
    main()

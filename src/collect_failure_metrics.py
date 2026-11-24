"""事后采集 4 类失效指标（依赖阶段2跑完的 30 例 trace + 阶段1的 evidence 标注）。

对每个 run（30例 × 4方法）做一次 run 级 DeepSeek 抽取，产出：
- ddx evolution：每轮医生的诊断假设（DDx 快照）
- evidence_status：每个关键证据的状态（not_asked / disclosed / used）
- anchored / correct_diagnosis_appeared / correct_diagnosis_dropped（锚定指标）
- repeated_questions（动作质量：重复提问）

产出 results/failure_metrics.json（120 个 run 的指标）。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import get_llm, chat

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
RESULTS = EID / "results" / "medqa"
EVID = json.loads(Path("/workspace/medical_llm_exp/data/evidence_annotation_30.json").read_text())
OUT = Path("/workspace/medical_llm_exp/results/failure_metrics.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

METHODS = ["roleplay", "refine", "meddxagent", "clinicalagents"]

SYSTEM = (
    "You are analyzing a diagnostic conversation between a doctor agent and a simulated patient. "
    "You are given the conversation trace, the correct diagnosis, and a list of key evidence.\n\n"
    "Tasks:\n"
    "1. Reconstruct the doctor's diagnostic HYPOTHESIS EVOLUTION: for each meaningful turn, list the "
    "top 1-3 diagnoses the doctor was considering (with rough confidence high/med/low), inferred from "
    "the doctor's questions and actions. Also give the doctor's INITIAL hypotheses (first turn) and "
    "FINAL hypotheses (just before the diagnosis).\n"
    "2. For EACH key evidence item, set its status:\n"
    '   - "not_asked": the doctor never asked about it and it was never revealed.\n'
    '   - "disclosed": it appeared in the patient/exam answers (whether or not the doctor asked).\n'
    '   - "used": the doctor explicitly relied on it in reaching the final diagnosis.\n'
    "3. Anchoring: did the doctor's leading hypothesis stay essentially UNCHANGED from start to end "
    "(never seriously reconsidered)? -> anchored true/false.\n"
    "4. Did the CORRECT diagnosis ever appear in the doctor's hypotheses, and if yes, was it later "
    "DROPPED in favor of something else?\n"
    "5. List any REPEATED questions (the doctor asking the same thing more than once).\n\n"
    'Output JSON only, exact structure:\n'
    '{"initial_hypotheses": ["..."], "final_hypotheses": ["..."], '
    '"evolution": [{"turn": 1, "hypotheses": ["..."]}], "anchored": false, '
    '"correct_diagnosis_appeared": false, "correct_diagnosis_dropped": false, '
    '"evidence_status": [{"fact": "...", "status": "not_asked"}], '
    '"repeated_questions": ["..."]}'
)


def trace_summary(trace, max_chars=6000):
    """把 trace 压成紧凑文本：每轮 role + 动作 + 内容截断。"""
    lines = []
    for i, t in enumerate(trace):
        role = t.get("role_id", "?")
        c = re.sub(r"\s+", " ", t.get("content", "") or "").strip()
        # 只保留医生侧 + 患者/报告者的关键内容
        if role in ("patient", "reporter"):
            c = c[:200]
        else:
            c = c[:400]
        lines.append(f"[{i}][{role}] {c}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def extract(client, model, case_id, diagnosis, trace, evidence):
    prompt = (
        f"Correct diagnosis: {diagnosis}\n\n"
        f"Key evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Conversation trace:\n{trace_summary(trace)}\n\n"
        "Analyze."
    )
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            raw = chat(client, model, msgs, temperature=0.0, max_tokens=2000, json_mode=True)
            if isinstance(raw, dict) and "evolution" in raw:
                return raw
            if isinstance(raw, str):
                s = raw.strip().strip("`").lstrip("json").strip()
                try:
                    parsed = json.loads(s)
                    if "evolution" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"  [extract] {case_id} 第{attempt+1}次失败: {e}", file=sys.stderr)
    return None


def main():
    client, model = get_llm("deepseek")
    # 断点续跑
    metrics = {}
    if OUT.exists():
        metrics = json.loads(OUT.read_text())

    total = len(METHODS) * 30
    done = 0
    for m in METHODS:
        rec_dir = RESULTS / m / "qwen-plus" / "10_turns" / "record"
        if not rec_dir.is_dir():
            continue
        for cid in [f"agentclinic_medqa_{i}" for i in range(30)]:
            key = f"{m}/{cid}"
            if key in metrics and metrics[key].get("evolution"):
                done += 1
                continue
            fn = rec_dir / f"{cid}.json"
            if not fn.exists():
                print(f"[缺] {key} 无 record，跳过")
                continue
            r = json.loads(fn.read_text())
            diag = r.get("ground_truth") or ""
            ev = EVID.get(cid, {}).get("key_evidence", [])
            res = extract(client, model, cid, diag, r.get("trace", []), ev)
            if res is None:
                print(f"[失败] {key}")
                continue
            metrics[key] = {
                "case_id": cid, "method": m,
                "ground_truth": diag,
                "prediction": r.get("prediction"),
                "metrics": res,
            }
            done += 1
            OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
            if done % 10 == 0:
                print(f"  进度 {done}/{total}")

    print(f"\n完成 {done}/{total} -> {OUT}")


if __name__ == "__main__":
    main()

"""为 AgentClinic MedQA 前 30 例标注"鉴别性关键证据集"（供失效分析命中矩阵使用）。

用 DeepSeek 对每个病例标注：
- 3-5 个锁定正确诊断的关键证据（附英文检测关键词，用于在 trace 里检测是否被问出/披露）
- 1-3 个易混淆诊断及鉴别点

产出 data/evidence_annotation_30.json，支持断点续跑。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import get_llm, chat

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
DATA = BASE / "data" / "agentclinic_medqa_segmented.jsonl"
OUT = Path("/workspace/medical_llm_exp/data/evidence_annotation_30.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are an expert clinician building a diagnostic-evaluation dataset. "
    "Given a clinical case and its correct diagnosis, annotate KEY DISCRIMINATIVE evidence.\n\n"
    "Requirements:\n"
    "1. Identify 3-5 KEY pieces of evidence that SPECIFICALLY point to the correct diagnosis "
    "(NOT generic facts like age or sex). Prefer evidence that discriminates the correct diagnosis "
    "from its common confusable mimics.\n"
    "2. For each key evidence give:\n"
    "   - fact: the evidence as stated in the case, OR the specific clinical question a doctor "
    "     SHOULD ask to elicit it.\n"
    "   - type: \"patient\" (from history/symptoms) or \"exam\" (from physical exam/lab/imaging).\n"
    "   - detect_keywords: 2-5 English keywords/phrases that would appear in a doctor's questions "
    "     OR the patient's/exam answers if this evidence were actually elicited.\n"
    "   - why: one sentence on why this evidence is discriminative.\n"
    "3. List 1-3 CONFUSABLE diagnoses (common mimics) and, for each, the key distinguishing point.\n\n"
    'Output JSON only, with this exact structure:\n'
    '{"diagnosis": "...", "key_evidence": [{"fact": "...", "type": "patient", '
    '"detect_keywords": ["..."], "why": "..."}], "confusable": [{"disease": "...", '
    '"distinguishing": "..."}]}'
)


def build_case_text(d):
    osce = d["OSCE_Examination"]
    parts = []
    pa = osce.get("Patient_Actor", {})
    for k in ["Demographics", "History", "Symptoms", "Past_Medical_History",
              "Social_History", "Review_of_Systems"]:
        v = pa.get(k)
        if v:
            parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    pe = osce.get("Physical_Examination_Findings")
    if pe:
        parts.append(f"Physical_Examination: {json.dumps(pe, ensure_ascii=False)}")
    tr = osce.get("Test_Results")
    if tr:
        parts.append(f"Test_Results: {json.dumps(tr, ensure_ascii=False)}")
    return "\n".join(parts)


def annotate(client, model, case_text, diagnosis):
    prompt = f"Case:\n{case_text}\n\nCorrect diagnosis: {diagnosis}\n\nAnnotate."
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            raw = chat(client, model, msgs, temperature=0.0, max_tokens=1500, json_mode=True)
            if isinstance(raw, dict) and "key_evidence" in raw:
                return raw
            # 非标准 JSON：尝试粗提取
            if isinstance(raw, str):
                s = raw.strip()
                if s.startswith("```"):
                    s = s.strip("`").lstrip("json").strip()
                try:
                    parsed = json.loads(s)
                    if "key_evidence" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"  [annotate] 第{attempt+1}次失败: {e}", file=sys.stderr)
    return None


def main():
    client, model = get_llm("deepseek")
    # 断点续跑
    done = {}
    if OUT.exists():
        done = json.loads(OUT.read_text())

    cases = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    cases = cases[:30]  # 前 30 例

    for d in cases:
        cid = d.get("id")
        if cid in done and done[cid].get("key_evidence"):
            continue
        diag = d["OSCE_Examination"].get("Correct_Diagnosis", "")
        case_text = build_case_text(d)
        res = annotate(client, model, case_text, diag)
        if res is None:
            print(f"[跳过] {cid} 标注失败")
            continue
        done[cid] = res
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(f"[ok] {cid} -> {diag} | 证据 {len(res.get('key_evidence', []))} 条, "
              f"易混淆 {len(res.get('confusable', []))} 个")

    print(f"\n完成，共 {len(done)} 例标注 -> {OUT}")


if __name__ == "__main__":
    main()

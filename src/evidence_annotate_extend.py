"""为 AgentClinic MedQA 扩展标注关键证据集（id 30-59），合并旧 30 例产出 60 例。

复用 evidence_annotate.py 的标注逻辑，支持断点续跑，最终产出
data/evidence_annotation_60.json（含 id 0-59）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_annotate import SYSTEM, build_case_text, annotate
from llm_client import get_llm

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
DATA = BASE / "data" / "agentclinic_medqa_segmented.jsonl"
OLD = Path("/workspace/medical_llm_exp/data/evidence_annotation_30.json")
OUT = Path("/workspace/medical_llm_exp/data/evidence_annotation_60.json")

START, END = 30, 60


def main():
    client, model = get_llm("deepseek")

    # 继承旧 30 例标注（若有）
    done = {}
    if OUT.exists():
        done = json.loads(OUT.read_text())
    elif OLD.exists():
        done = json.loads(OLD.read_text())

    cases = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    cases = cases[START:END]

    for d in cases:
        cid = d.get("id")
        if cid in done and done[cid].get("key_evidence"):
            continue
        diag = d["OSCE_Examination"].get("Correct_Diagnosis", "")
        case_text = build_case_text(d)
        res = annotate(client, model, case_text, diag)
        if res is None:
            print(f"[跳过] {cid} 标注失败", flush=True)
            continue
        done[cid] = res
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(f"[ok] {cid} -> {diag} | 证据 {len(res.get('key_evidence', []))} 条, "
              f"易混淆 {len(res.get('confusable', []))} 个", flush=True)

    print(f"\n完成，共 {len(done)} 例标注 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

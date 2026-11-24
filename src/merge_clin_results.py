"""合并 clinicalagents 的 3 个分片结果，重算 result.json。"""
import json
import shutil
from pathlib import Path

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
OUT_DIR = BASE / "results/medqa/clinicalagents/qwen-plus/10_turns"
RECORD_DIR = OUT_DIR / "record"
RECORD_DIR.mkdir(parents=True, exist_ok=True)

all_results = []
for shard in range(3):
    src_record = BASE / f"results_clin{shard}/medqa/clinicalagents/qwen-plus/10_turns/record"
    if not src_record.exists():
        print(f"shard {shard}: 无 record 目录，跳过")
        continue
    for f in src_record.glob("*.json"):
        shutil.copy2(f, RECORD_DIR / f.name)
        r = json.loads(f.read_text())
        all_results.append(r)

print(f"合并 {len(all_results)} 个病例 record")

# 重算汇总
results_out = []
correct = 0
icrs, pcovs, ecovs = [], [], []
for r in all_results:
    cov = r.get("coverage") or {}
    icr = cov.get("information_coverage_rate")
    pcov = cov.get("patient_coverage")
    ecov = cov.get("exam_coverage")
    if r.get("is_correct"):
        correct += 1
    if icr is not None:
        icrs.append(icr)
    if pcov is not None:
        pcovs.append(pcov)
    if ecov is not None:
        ecovs.append(ecov)
    results_out.append({
        "case_id": r.get("case_id"),
        "prediction": r.get("prediction"),
        "ground_truth": r.get("ground_truth"),
        "is_correct": r.get("is_correct"),
        "judge_response": r.get("judge_response"),
        "information_coverage_rate": icr,
        "patient_coverage": pcov,
        "exam_coverage": ecov,
    })

n = len(all_results)
summary = {
    "total_items": n,
    "correct_count": correct,
    "top1_accuracy": correct / n if n else 0,
    "avg_information_coverage_rate": sum(icrs) / len(icrs) if icrs else 0,
    "avg_patient_coverage": sum(pcovs) / len(pcovs) if pcovs else 0,
    "avg_exam_coverage": sum(ecovs) / len(ecovs) if ecovs else 0,
    "results": results_out,
}
(OUT_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print(f"重算完成: total={n}, correct={correct}, acc={summary['top1_accuracy']:.3f}, ICR={summary['avg_information_coverage_rate']:.3f}")

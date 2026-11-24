"""冒烟测试：验证 refine_cv_qwen（同模型核对）与 refine_sc（self-consistency）能跑通。"""
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)
sys.path.insert(0, str(EID / "src"))
sys.path.insert(0, "/workspace/medical_llm_exp/src")

from dotenv import load_dotenv
load_dotenv(EID / ".env")

from eid.config import ModelConfig
from eid.datasets import load_dataset
from eid.scenarios import get_scenario
from eid.scenarios.base import CaseInput

CASE_IDS = ["agentclinic_medqa_14"]


def run_one(mode: str):
    dataset = load_dataset("medqa")
    items = {it.case_id: it for it in dataset}
    doctor = ModelConfig.from_string("qwen-plus")
    scen = get_scenario(mode=mode, dataset_name="medqa", doctor_config=doctor, max_turns=10)
    for cid in CASE_IDS:
        case = items[cid]
        print(f"\n===== {mode} / {cid}  truth={case.answer!r} =====", flush=True)
        case_input = CaseInput(
            case_id=case.case_id,
            task=case.task,
            patient_facts=case.patient_facts,
            exam_facts=case.exam_facts,
            ground_truth=case.answer,
        )
        res = scen.run(case_input)
        print(f"  answer = {res.answer!r}", flush=True)
        for t in res.trace:
            if t.get("role_id") == "diagnostician_sc":
                print(f"  SC voted = {t['content'].get('voted')!r}", flush=True)
                print(f"  SC samples = {[s[:60] for s in t['content'].get('samples', [])]}", flush=True)
        print("  OK", flush=True)


if __name__ == "__main__":
    for m in ["refine_cv_qwen", "refine_sc"]:
        try:
            run_one(m)
        except Exception as e:
            import traceback
            print(f"[FAIL] {m}: {e}", flush=True)
            traceback.print_exc()

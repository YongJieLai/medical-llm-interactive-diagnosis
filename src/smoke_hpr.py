"""HPR 插件冒烟测试：跑 refine 失效的 3 个"正确假设被丢弃"case，看 HPR 是否改对。"""
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)
sys.path.insert(0, str(EID / "src"))

from dotenv import load_dotenv
load_dotenv(EID / ".env")

from eid.config import ModelConfig
from eid.datasets import load_dataset
from eid.scenarios import get_scenario

TARGET = ["agentclinic_medqa_14", "agentclinic_medqa_16", "agentclinic_medqa_15"]


def main():
    dataset = load_dataset("medqa")
    items = [it for it in dataset if it.case_id in TARGET]
    items.sort(key=lambda it: TARGET.index(it.case_id))

    doctor = ModelConfig.from_string("qwen-plus")
    scenario = get_scenario(mode="refine_hpr", dataset_name="medqa",
                            doctor_config=doctor, max_turns=10)

    for it in items:
        print("=" * 80)
        print(f"case={it.case_id}  truth={it.answer!r}")
        r = scenario.run(__import__("eid.scenarios.base", fromlist=["CaseInput"]).CaseInput(
            case_id=it.case_id,
            task=it.task,
            patient_facts=it.patient_facts,
            exam_facts=it.exam_facts,
            ground_truth=it.answer,
        ))
        print(f"answer={r.answer!r}")
        print(f"turns={r.metadata.get('turns')}")
        # 打印 differential 与 diagnostician 的关键输出
        for t in r.trace:
            if t["role_id"] == "differential":
                print("\n--- differential [DDX] ---")
                print(t["content"][:1200])
            elif t["role_id"] == "diagnostician":
                print("\n--- diagnostician ---")
                print(t["content"][:900])
            elif t["role_id"] == "diagnostician_verifier":
                print("\n--- verifier ---")
                print(t["content"][:600])
        print()


if __name__ == "__main__":
    main()

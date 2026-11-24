"""冒烟测试：3 个关键失效 case 验证 refine_cv 跑通 + 跨模型 verifier JSON + 规则树决策。"""
import os
import sys
import json
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


def main():
    dataset = load_dataset("medqa")
    target = {"agentclinic_medqa_14", "agentclinic_medqa_15", "agentclinic_medqa_16"}
    items = [it for it in dataset if it.case_id in target]
    print(f"命中 {len(items)} 个 case")

    doctor = ModelConfig.from_string("qwen-plus")
    scenario = get_scenario(mode="refine_cv", dataset_name="medqa",
                            doctor_config=doctor, max_turns=10)

    for it in items:
        r = scenario.run(it)
        print("=" * 80)
        print(f"{it.case_id}  truth={getattr(it, 'ground_truth', '')!r}")
        print(f"  answer={r.answer[:120]!r}")
        for t in r.trace:
            if t["role_id"] == "verifier":
                print("  [verifier]", json.dumps(t["content"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()

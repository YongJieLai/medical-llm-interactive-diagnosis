"""深挖单个 case 的完整 trace：医生动作 + summary + diagnostician + verifier 完整 json。"""
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
    items = {it.case_id: it for it in dataset}
    it = items["agentclinic_medqa_14"]
    print(f"truth = {it.answer!r}\n")

    doctor = ModelConfig.from_string("qwen-plus")
    scenario = get_scenario(mode="refine_cv", dataset_name="medqa",
                            doctor_config=doctor, max_turns=10)
    r = scenario.run(it)

    print(f"FINAL ANSWER = {r.answer!r}\n")
    print("=" * 90)
    for t in r.trace:
        rid = t["role_id"]
        c = t["content"]
        if rid == "verifier":
            # 完整 json
            print(f"\n### [verifier] 完整输出 ###")
            print(json.dumps(c, ensure_ascii=False, indent=1))
        elif rid == "doctor":
            # 只打印动作
            import re
            m = re.search(r"\[(QUERY|TEST|FINISH|DIAGNOSIS)\]\s*(.*)", c, re.S)
            if m:
                print(f"\n--- [doctor] {m.group(1)} ---")
                print(m.group(2).strip()[:300])
            else:
                print(f"\n--- [doctor] ---\n{c[:200]}")
        elif rid == "summarizer":
            m = re.search(r"\[SUMMARY\]\s*([\s\S]*)", c)
            s = m.group(1) if m else c
            print(f"\n### [summarizer] ###\n{s[:600]}")
        elif rid == "diagnostician":
            print(f"\n### [diagnostician] 完整输出 ###")
            print(c[:1200])
        elif rid == "patient":
            m = re.search(r"\[RESPONSE\]\s*([\s\S]*)", c)
            s = m.group(1) if m else c
            print(f"\n--- [patient] ---\n{s[:200]}")
        elif rid == "reporter":
            print(f"\n--- [reporter] ---\n{c[:200]}")
    print()


if __name__ == "__main__":
    main()

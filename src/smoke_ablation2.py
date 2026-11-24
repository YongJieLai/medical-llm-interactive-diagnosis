"""冒烟测试：验证 3 个消融 + qwenmax + 全 deepseek 变体能跑通。

先测 qwen-max 连通性，再逐个跑 medqa_14（refine_cv_qwen 曾修好该例）。
"""
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)
sys.path.insert(0, str(EID / "src"))
sys.path.insert(0, "/workspace/medical_llm_exp/src")

from dotenv import load_dotenv
load_dotenv(EID / ".env")
from llm_client import _load_env_file, get_llm, chat as _chat
_load_env_file()

from eid.config import ModelConfig
from eid.datasets import load_dataset
from eid.scenarios import get_scenario
from eid.scenarios.base import CaseInput


def build_doctor(provider):
    if provider == "deepseek":
        return ModelConfig.from_string(
            "deepseek-chat",
            api_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    return ModelConfig.from_string(provider)


def test_qwenmax():
    try:
        client, _ = get_llm("qwen")
        r = _chat(client, "qwen-max", [{"role": "user", "content": "回复: ok"}], max_tokens=8)
        print(f"[qwen-max] 连通 OK -> {r!r}")
        return True
    except Exception as e:
        print(f"[qwen-max] 不可用: {e}")
        return False


def main():
    ok_max = test_qwenmax()
    dataset = load_dataset("medqa")
    items = {it.case_id: it for it in dataset}
    item = items["agentclinic_medqa_14"]
    case = CaseInput(
        case_id=item.case_id,
        task=item.task,
        patient_facts=item.patient_facts,
        exam_facts=item.exam_facts,
        ground_truth=item.answer,
    )
    print(f"truth = {item.answer!r}")

    variants = [
        ("refine_cv_qwen_check", "qwen-plus"),
        ("refine_cv_qwen_gap", "qwen-plus"),
        ("refine_cv_qwen_refute", "qwen-plus"),
    ]
    if ok_max:
        variants.append(("refine_cv_qwenmax", "qwen-plus"))
    variants.append(("refine_cv", "deepseek"))

    for mode, doc in variants:
        try:
            scen = get_scenario(
                mode=mode, dataset_name="medqa",
                doctor_config=build_doctor(doc), max_turns=10,
            )
            res = scen.run(case)
            print(f"[{mode}/{doc}] answer={res.answer[:80]!r}", flush=True)
        except Exception:
            import traceback
            print(f"[{mode}/{doc}] ERROR:", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()

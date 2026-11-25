"""P2 层 2 全量：跑 RefineCVQwenGapRefuteLLMDDXScenario（运行时 LLM 生成鉴别诊断）30 例。

与 baseline `refine_cv_qwen_gap_refute`（同父类、无 RAG）同口径：
  - qwen-plus 医生/病人/测量，verifier 同 qwen
  - max_turns=10，dataset 前 30 例
  - 反证 grounding 用运行时 LLM（DeepSeek）生成候选病的鉴别诊断+区分点（不泄题）

产出 record：
  benchmarks/eid-benchmark/results/medqa/refine_cv_qwen_gap_refute_llmddx/qwen-plus/10_turns/record/*.json

运行：python3 src/run_refine_llmddx.py [N=30] [max_turns=10]
"""
import os
import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EID_ROOT = REPO / "benchmarks" / "eid-benchmark"

_env = {}
for line in (REPO / "config" / "api_keys.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        _env[k.strip()] = v.strip()

DASHSCOPE_KEY = _env.get("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_KEY:
    print("缺少 DASHSCOPE_API_KEY")
    sys.exit(1)

os.environ.setdefault("OPENAI_API_KEY", DASHSCOPE_KEY)
os.environ.setdefault(
    "OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EID_ROOT / "src"))
os.chdir(EID_ROOT)

from eid.config import ModelConfig  # noqa: E402
from eid.datasets import load_dataset  # noqa: E402
from eid.scenarios.base import CaseInput  # noqa: E402
from eid.scenarios.refine_cv_runtimeddx import (  # noqa: E402
    RefineCVQwenGapRefuteLLMDDXScenario,
)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    out_dir = (
        EID_ROOT / "results" / "medqa" / "refine_cv_qwen_gap_refute_llmddx"
        / "qwen-plus" / f"{max_turns}_turns" / "record"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("medqa")
    items = list(dataset)[:n]
    cfg = ModelConfig.from_string("qwen-plus")

    print(f"=== 运行时 LLM DDx grounding 全量：n={len(items)}, max_turns={max_turns} ===\n", flush=True)
    t_start = time.time()
    for i, item in enumerate(items, 1):
        scenario = RefineCVQwenGapRefuteLLMDDXScenario(
            dataset_name="medqa",
            doctor_config=cfg,
            patient_config=cfg,
            reporter_config=cfg,
            max_turns=max_turns,
        )
        ci = CaseInput(
            case_id=item.case_id,
            task=item.task,
            patient_facts=item.patient_facts,
            exam_facts=item.exam_facts,
            ground_truth=item.answer,
        )
        t0 = time.time()
        try:
            result = scenario.run(ci)
            rec = {
                "case_id": item.case_id,
                "prediction": result.answer,
                "ground_truth": item.answer,
                "trace": result.trace,
                "metadata": result.metadata,
            }
        except Exception as e:  # noqa: BLE001
            rec = {
                "case_id": item.case_id,
                "prediction": "",
                "ground_truth": item.answer,
                "error": str(e),
                "trace": [],
            }
            print(f"[{i}/{len(items)}] {item.case_id} ERROR: {e}", flush=True)
        else:
            fn = out_dir / f"{item.case_id}.json"
            fn.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
            print(
                f"[{i}/{len(items)}] {item.case_id:24s} "
                f"answer={result.answer[:50]!r:52s} {time.time()-t0:.0f}s",
                flush=True,
            )

    print(f"\n完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()

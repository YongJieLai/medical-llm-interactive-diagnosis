"""在 30 例（id 0-29）上跑 refine_hpr，输出到 results/medqa/refine_hpr/qwen-plus/10_turns/。

- 用 ExactMatchMetric 占位（不调 LLM judge），is_correct 事后用确定性 judge 重判。
- record 按 case_id 命名保存，供事后 rejudge / 失效指标采集复用。
"""
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)
sys.path.insert(0, str(EID / "src"))

from dotenv import load_dotenv
load_dotenv(EID / ".env")

from eid.config import ModelConfig
from eid.datasets import load_dataset, Dataset
from eid.metrics import ExactMatchMetric
from eid.scenarios import get_scenario
from eid.benchmark import Benchmark

OUT_BASE = EID / "results" / "medqa"


def main():
    dataset = load_dataset("medqa")
    items = list(dataset)[0:30]
    sub = Dataset(items=items, name="medqa")
    print(f"将跑 {len(sub)} 例: {[it.case_id for it in sub][:5]} ... {[it.case_id for it in sub][-3:]}")

    doctor = ModelConfig.from_string("qwen-plus")
    scenario = get_scenario(mode="refine_hpr", dataset_name="medqa",
                            doctor_config=doctor, max_turns=10)
    out_dir = OUT_BASE / "refine_hpr" / "qwen-plus" / "10_turns"
    metric = ExactMatchMetric()
    bm = Benchmark(dataset=sub, metric=metric, output_dir=out_dir, save_traces=True)
    summary = bm.evaluate(
        scenario=scenario,
        max_workers=3,
        summary_path=out_dir / "result_new.json",
    )
    print(f"[done] refine_hpr: acc={summary['top1_accuracy']:.3f} "
          f"({summary['correct_count']}/{summary['total_items']})")


if __name__ == "__main__":
    main()

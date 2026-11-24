"""在 30 例（id 0-29）上跑 refine_cv（跨模型证据核对 + 缺口驱动反证）。"""
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
from eid.datasets import load_dataset, Dataset
from eid.metrics import ExactMatchMetric
from eid.scenarios import get_scenario
from eid.benchmark import Benchmark

OUT_BASE = EID / "results" / "medqa"


def main():
    dataset = load_dataset("medqa")
    items = list(dataset)[0:30]
    sub = Dataset(items=items, name="medqa")
    print(f"将跑 {len(sub)} 例 refine_cv")

    doctor = ModelConfig.from_string("qwen-plus")
    scenario = get_scenario(mode="refine_cv", dataset_name="medqa",
                            doctor_config=doctor, max_turns=10)
    out_dir = OUT_BASE / "refine_cv" / "qwen-plus" / "10_turns"
    metric = ExactMatchMetric()
    bm = Benchmark(dataset=sub, metric=metric, output_dir=out_dir, save_traces=True)
    summary = bm.evaluate(
        scenario=scenario,
        max_workers=3,
        summary_path=out_dir / "result_new.json",
    )
    print(f"[done] refine_cv: exactmatch_acc={summary['top1_accuracy']:.3f} "
          f"({summary['correct_count']}/{summary['total_items']})  (占位，最终看确定性 judge)")


if __name__ == "__main__":
    main()

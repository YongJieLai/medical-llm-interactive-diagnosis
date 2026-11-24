"""跑新增 15 例（case_id 15-29），4 方法 × 10_turns，doctor 用 qwen-plus。

- 只跑 id 15-29（新病例），复用 id 0-14 已有结果。
- 跑的时候用 ExactMatchMetric 占位（不调 LLM judge），is_correct 事后用确定性 judge 统一重判。
- record 写入 results/medqa/{mode}/qwen-plus/10_turns/record/（按 case_id 命名，与旧结果不冲突）。
- summary 写 result_new.json（不覆盖旧 result.json）。
"""
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)  # 数据/结果路径相对 EID 目录
sys.path.insert(0, str(EID / "src"))

from dotenv import load_dotenv
load_dotenv(EID / ".env")

from eid.config import ModelConfig
from eid.datasets import load_dataset, Dataset
from eid.metrics import ExactMatchMetric
from eid.scenarios import get_scenario
from eid.benchmark import Benchmark

MODES = ["roleplay", "refine", "meddxagent", "clinicalagents"]
OUT_BASE = EID / "results" / "medqa"


def main():
    dataset = load_dataset("medqa")            # 200 条
    items = list(dataset)[15:30]               # 新 15 例（id 15-29）
    sub = Dataset(items=items, name="medqa")
    print(f"将跑 {len(sub)} 例: {[it.case_id for it in sub]}")

    doctor = ModelConfig.from_string("qwen-plus")

    for mode in MODES:
        scenario = get_scenario(
            mode=mode, dataset_name="medqa",
            doctor_config=doctor, max_turns=10,
        )
        out_dir = OUT_BASE / mode / "qwen-plus" / "10_turns"
        metric = ExactMatchMetric()            # 占位，事后重判
        bm = Benchmark(dataset=sub, metric=metric, output_dir=out_dir, save_traces=True)
        summary = bm.evaluate(
            scenario=scenario,
            max_workers=3,
            summary_path=out_dir / "result_new.json",
        )
        print(f"[done] {mode}: acc={summary['top1_accuracy']:.3f} "
              f"({summary['correct_count']}/{summary['total_items']})")


if __name__ == "__main__":
    main()

"""通用跑批脚本：跑一个变体（30 例，id 0-29）。

用法：
  python3 src/run_variants.py refine_cv_qwen_check
  python3 src/run_variants.py refine_cv_qwen_gap
  python3 src/run_variants.py refine_cv_qwen_refute
  python3 src/run_variants.py refine_cv_qwenmax
  python3 src/run_variants.py refine_cv --doctor deepseek   # 全 DeepSeek 同模型

--doctor 决定生成端（doctor/summarizer/diagnostician/patient/reporter）模型，
verifier 由场景类自身的 VERIFIER_PROVIDER/VERIFIER_MODEL 决定。
"""
import argparse
import os
import sys
from pathlib import Path

EID = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark")
os.chdir(EID)
sys.path.insert(0, str(EID / "src"))
sys.path.insert(0, "/workspace/medical_llm_exp/src")

from dotenv import load_dotenv
load_dotenv(EID / ".env")

# 加载 api_keys.env（提供 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY）
from llm_client import _load_env_file
_load_env_file()

from eid.config import ModelConfig
from eid.datasets import load_dataset, Dataset
from eid.metrics import ExactMatchMetric
from eid.scenarios import get_scenario
from eid.benchmark import Benchmark

OUT_BASE = EID / "results" / "medqa"


def build_doctor(provider: str) -> ModelConfig:
    if provider == "deepseek":
        return ModelConfig.from_string(
            "deepseek-chat",
            api_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    if provider == "glm":
        return ModelConfig.from_string(
            os.environ.get("GLM_MODEL", "glm-4-flash"),
            api_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=os.environ.get("ZHIPU_API_KEY"),
        )
    # qwen-plus / qwen-max 等走 DashScope OpenAI 兼容端点（显式传 key，不依赖 .env）
    return ModelConfig.from_string(
        provider,
        api_url=os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", type=str)
    ap.add_argument("--doctor", default="qwen-plus")
    ap.add_argument("--out", default=None, help="输出子目录名（默认=mode）")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="起始病例下标（含）")
    ap.add_argument("--end", type=int, default=30, help="结束病例下标（不含）")
    ap.add_argument("--dataset-path", default=None, help="自定义数据集 jsonl 路径（默认用 medqa 默认路径）")
    args = ap.parse_args()

    dataset = load_dataset("medqa", path=args.dataset_path)
    items = list(dataset)[args.start:args.end]
    sub = Dataset(items=items, name="medqa")
    print(f"[{args.mode}] 将跑 {len(sub)} 例 (doctor={args.doctor}, range={args.start}:{args.end})", flush=True)

    doctor = build_doctor(args.doctor)
    scenario = get_scenario(
        mode=args.mode,
        dataset_name="medqa",
        doctor_config=doctor,
        max_turns=args.max_turns,
    )
    out_name = args.out or args.mode
    out_dir = OUT_BASE / out_name / args.doctor / f"{args.max_turns}_turns"
    metric = ExactMatchMetric()
    bm = Benchmark(dataset=sub, metric=metric, output_dir=out_dir, save_traces=True)
    summary = bm.evaluate(
        scenario=scenario,
        max_workers=args.workers,
        summary_path=out_dir / "result_new.json",
    )
    print(
        f"[done] {args.mode}: exactmatch={summary['top1_accuracy']:.3f} "
        f"({summary['correct_count']}/{summary['total_items']})  (占位，最终看确定性 judge)",
        flush=True,
    )


if __name__ == "__main__":
    main()

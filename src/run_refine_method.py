"""通用 runner：跑指定的 refine_cv_qwen 变体 30 例（同口径）。

支持 method：
  - rag        : RefineCVQwenGapRefuteRAGScenario  （oracle KB，实验② 同批重跑）
  - emptyddx   : RefineCVQwenGapRefuteEmptyDDXScenario（空 DDx 占位，实验③方案1）
  - doctorddx  : RefineCVQwenGapRefuteDoctorDDXScenario（P3 DDx 驱动问诊）
  - ddxdiag    : RefineCVQwenGapRefuteDDXDiagScenario（P4 方案2a diagnostician 两步法）
  - neutralsumm: RefineCVQwenGapRefuteNeutralSummScenario（P4 方案2b summarizer 中性转录）
  - joint      : RefineCVQwenGapRefuteJointScenario（P4 方案2c 联合）
  - gaprefute  : RefineCVQwenGapRefuteScenario（baseline 同批重跑，输出到 _rerun 目录，消除时间漂移）
  - llmddxns   : RefineCVQwenGapRefuteLLMDDXNeutralSummScenario（层2 + 2b 组合）
  - llmddxqwen : RefineCVQwenGapRefuteLLMDDXQwenGenScenario（层2 DDx 生成换 qwen-plus）

与 baseline refine_cv_qwen_gap_refute 同口径：
  - qwen-plus 医生/病人/测量/总结/诊断，verifier 用 DeepSeek
  - max_turns=10，dataset 前 30 例（case_id 与 baseline/层2 完全一致）

运行：python3 src/run_refine_method.py <method> 30 10
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
from eid.scenarios.refine_cv_ablations import (  # noqa: E402
    RefineCVQwenGapRefuteRAGScenario,
    RefineCVQwenGapRefuteScenario,
)
from eid.scenarios.refine_cv_llmddx_variants import (  # noqa: E402
    RefineCVQwenGapRefuteLLMDDXNeutralSummScenario,
    RefineCVQwenGapRefuteLLMDDXQwenGenScenario,
)
from eid.scenarios.refine_clean import RefineCleanScenario  # noqa: E402
from eid.scenarios.refine_cv_runtimeddx import RefineCVQwenGapRefuteLLMDDXScenario  # noqa: E402
from eid.scenarios.refine_cv_emptyddx import RefineCVQwenGapRefuteEmptyDDXScenario  # noqa: E402
from eid.scenarios.refine_cv_doctorddx import RefineCVQwenGapRefuteDoctorDDXScenario  # noqa: E402
from eid.scenarios.refine_cv_ddxdiag import (  # noqa: E402
    RefineCVQwenGapRefuteDDXDiagScenario,
    RefineCVQwenGapRefuteNeutralSummScenario,
    RefineCVQwenGapRefuteJointScenario,
)

METHODS = {
    "rag": ("refine_cv_qwen_gap_refute_rag", RefineCVQwenGapRefuteRAGScenario),
    "emptyddx": ("refine_cv_qwen_gap_refute_emptyddx", RefineCVQwenGapRefuteEmptyDDXScenario),
    "doctorddx": ("refine_cv_qwen_gap_refute_doctorddx", RefineCVQwenGapRefuteDoctorDDXScenario),
    "ddxdiag": ("refine_cv_qwen_gap_refute_ddxdiag", RefineCVQwenGapRefuteDDXDiagScenario),
    "neutralsumm": ("refine_cv_qwen_gap_refute_neutralsumm", RefineCVQwenGapRefuteNeutralSummScenario),
    "joint": ("refine_cv_qwen_gap_refute_joint", RefineCVQwenGapRefuteJointScenario),
    "gaprefute": ("refine_cv_qwen_gap_refute_rerun", RefineCVQwenGapRefuteScenario),
    "llmddx": ("refine_cv_qwen_gap_refute_llmddx_rerun", RefineCVQwenGapRefuteLLMDDXScenario),
    "llmddxns": ("refine_cv_qwen_gap_refute_llmddx_neutralsumm", RefineCVQwenGapRefuteLLMDDXNeutralSummScenario),
    "llmddxqwen": ("refine_cv_qwen_gap_refute_llmddx_qwengen", RefineCVQwenGapRefuteLLMDDXQwenGenScenario),
    # 干净 baseline：上游 RefineScenario 原样（PASS/INCOMPLETE verifier，同模型 verifier）
    "refineclean": ("refine_clean", RefineCleanScenario),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in METHODS:
        print(f"用法: python3 {sys.argv[0]} <{'|'.join(METHODS)}> [N=30] [max_turns=10]")
        sys.exit(2)
    method = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_turns = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dirname, cls = METHODS[method]

    out_dir = (
        EID_ROOT / "results" / "medqa" / dirname
        / "qwen-plus" / f"{max_turns}_turns" / "record"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("medqa")
    items = list(dataset)[:n]
    cfg = ModelConfig.from_string("qwen-plus")

    print(f"=== {method} 全量：n={len(items)}, max_turns={max_turns} ===\n", flush=True)
    t_start = time.time()
    for i, item in enumerate(items, 1):
        fn = out_dir / f"{item.case_id}.json"
        if fn.exists() and fn.stat().st_size > 0:
            print(f"[{i}/{len(items)}] {item.case_id} SKIP (已存在)", flush=True)
            continue
        scenario = cls(
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

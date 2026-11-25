"""P1 全量验证：在真实环境跑 RefineHTScenario（HypothesisTracker 记忆增强）30 例。

与 baseline `refine_cv_qwen`（同父类、无 HT、全开关）同口径：
  - qwen-plus 医生/病人/测量，verifier 同 qwen（RefineCVQwenScenario）
  - max_turns=10，dataset 前 30 例（agentclinic_medqa_0..29）
  - 每例新建 scenario 实例（避免 HypothesisTracker 跨例/跨线程竞态）
  - 串行执行

产出 record：benchmarks/eid-benchmark/results/medqa/refine_ht/qwen-plus/10_turns/record/*.json
   record 结构 = {case_id, prediction, ground_truth, trace, metadata}，与 Benchmark 输出一致。

运行（key 从 config/api_keys.env 读，不入库）：
  python3 src/run_refine_ht.py [N=30] [max_turns=10]
"""
import os
import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # 老仓库根
EID_ROOT = REPO / "benchmarks" / "eid-benchmark"

# 1) 从 config/api_keys.env 读 DASHSCOPE key（不入库）
_env = {}
for line in (REPO / "config" / "api_keys.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        _env[k.strip()] = v.strip()

DASHSCOPE_KEY = _env.get("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_KEY:
    print("缺少 DASHSCOPE_API_KEY，请在 config/api_keys.env 配置")
    sys.exit(1)

# 2) camel agent 走 OpenAI 兼容端点（DashScope）
os.environ.setdefault("OPENAI_API_KEY", DASHSCOPE_KEY)
os.environ.setdefault(
    "OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 3) 路径
sys.path.insert(0, str(REPO / "src"))       # llm_client
sys.path.insert(0, str(EID_ROOT / "src"))   # eid 包
os.chdir(EID_ROOT)                          # dataset 相对路径 data/...

from eid.config import ModelConfig                      # noqa: E402
from eid.datasets import load_dataset                   # noqa: E402
from eid.scenarios.base import CaseInput                # noqa: E402
from eid.scenarios.refine_ht import RefineHTScenario    # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    out_dir = (
        EID_ROOT / "results" / "medqa" / "refine_ht"
        / "qwen-plus" / f"{max_turns}_turns" / "record"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("medqa")
    items = list(dataset)[:n]
    cfg = ModelConfig.from_string("qwen-plus")

    print(f"=== RefineHT 全量验证：n={len(items)}, max_turns={max_turns} ===\n", flush=True)
    t_start = time.time()
    for i, item in enumerate(items, 1):
        # 每例新建 scenario，隔离 HypothesisTracker（跨例不泄漏、无竞态）
        scenario = RefineHTScenario(
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
        except Exception as e:  # noqa: BLE001 - 单例失败不中断整批
            rec = {
                "case_id": item.case_id,
                "prediction": "",
                "ground_truth": item.answer,
                "error": str(e),
                "trace": [],
            }
            print(f"[{i}/{len(items)}] {item.case_id} ERROR: {e}", flush=True)
        else:
            # tracker 状态摘要（验证机制在线生效）
            tracker_records = [
                t for t in result.trace if t.get("role_id") == "hypothesis_tracker"
            ]
            max_hyp = 0
            for t in tracker_records:
                c = t.get("content")
                if isinstance(c, dict):
                    max_hyp = max(max_hyp, len(c.get("hypotheses", [])))
            fn = out_dir / f"{item.case_id}.json"
            fn.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
            print(
                f"[{i}/{len(items)}] {item.case_id:24s} "
                f"answer={result.answer[:50]!r:52s} "
                f"tracker_records={len(tracker_records)} max_hyp={max_hyp} "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )

    print(f"\n完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()

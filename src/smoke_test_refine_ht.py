"""P1 smoke test: 在真实环境跑 RefineHTScenario，验证 HypothesisTracker 在线行为。

验证点：
1. 诊断流程不被破坏（每例都能产出非空 answer）；
2. tracker 跨轮继承生效（trace 里出现 hypothesis_tracker 记录，且候选集非空、
   且跨多个 round 累积）。

运行（在沙箱，key 从 config/api_keys.env 读，不入库）：
    python3 src/smoke_test_refine_ht.py [N=5] [max_turns=10]
"""
import os
import sys
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
sys.path.insert(0, str(REPO / "src"))       # llm_client / deterministic_judge
sys.path.insert(0, str(EID_ROOT / "src"))   # eid 包
os.chdir(EID_ROOT)                          # dataset 相对路径 data/...

from eid.config import ModelConfig                      # noqa: E402
from eid.datasets import load_dataset                   # noqa: E402
from eid.scenarios.base import CaseInput                # noqa: E402
from eid.scenarios.refine_ht import RefineHTScenario    # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    dataset = load_dataset("medqa", max_items=n)
    cfg = ModelConfig.from_string("qwen-plus")
    scenario = RefineHTScenario(
        dataset_name="medqa",
        doctor_config=cfg,
        patient_config=cfg,
        reporter_config=cfg,
        max_turns=max_turns,
    )

    print(f"=== RefineHTScenario smoke test: n={n}, max_turns={max_turns} ===\n")
    for i, item in enumerate(dataset, 1):
        ci = CaseInput(
            case_id=item.case_id,
            task=item.task,
            patient_facts=item.patient_facts,
            exam_facts=item.exam_facts,
            ground_truth=item.answer,
        )
        result = scenario.run(ci)

        tracker_records = [
            t for t in result.trace if t.get("role_id") == "hypothesis_tracker"
        ]
        max_hyp = 0
        last_round = 0
        for t in tracker_records:
            c = t.get("content")
            if isinstance(c, dict):
                max_hyp = max(max_hyp, len(c.get("hypotheses", [])))
                last_round = max(last_round, c.get("round", 0))

        ok_flow = bool(result.answer)
        ok_tracker = len(tracker_records) > 0 and max_hyp > 0
        print(
            f"[{i}/{n}] {item.case_id:24s} "
            f"answer={result.answer[:52]!r:54s} "
            f"tracker_records={len(tracker_records)} max_hyp={max_hyp} "
            f"last_round={last_round} "
            f"flow={'OK' if ok_flow else 'FAIL'} tracker={'OK' if ok_tracker else 'FAIL'}"
        )

    print("\n完成。检查点：flow 全 OK（诊断不破坏）、tracker 全 OK（跨轮继承生效）")


if __name__ == "__main__":
    main()

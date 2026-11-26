"""跑论文 EID-Benchmark 原文的 5 种模式，max_turns=16，MedQA 200 例，case 级并行。

论文方法对照（arXiv:2601.19773, "Strong Reasoning Isn't Enough"）：
  cot      -> Upper Bound（静态全信息，单次 CoT 推理）
  roleplay -> Baseline（标准交互式问诊）
  react    -> ReAct（显式 Thought-Act）
  sc       -> Summarized-Conversation（先采集后总结诊断）
  refine   -> REFINE（SC + Diagnosis Verifier 门控，论文方法）

与论文实验设置对齐：max_turns=16、AgentClinic-MedQA 200 例、全角色同模型。
我们只有 DashScope key，故 doctor/patient/reporter/summarizer/diagnostician/verifier
统一用 qwen-plus（论文用 gpt-5-mini，模型名不同但角色结构一致）。

用法：
  python3 src/run_paper_modes.py <mode> [N=200] [max_turns=16] [workers=10]
  mode: cot|roleplay|react|sc|refine|all

输出目录：
  benchmarks/eid-benchmark/results/medqa/<mode>/qwen-plus/<turns>_turns/record/<case_id>.json
每个 json 含 prediction / ground_truth / trace / metadata；判分统一走 deterministic_judge.py。
"""
import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parent.parent
EID_ROOT = REPO / "benchmarks" / "eid-benchmark"

# ---------- 环境配置（读 DashScope key） ----------
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
from eid.scenarios import get_scenario, CoTScenario  # noqa: E402
from eid.scenarios.refine_clean import RefineCleanScenario  # noqa: E402
from eid.scenarios.refine_cv import RefineCVScenario  # noqa: E402
from eid.scenarios.refine_cv_ports import ReactCVScenario, RoleplayCVScenario  # noqa: E402
from eid.scenarios.react_cv2 import ReactCV2Scenario  # noqa: E402
from eid.scenarios.refine_cv_qwen import RefineCVQwenScenario  # noqa: E402
from eid.scenarios.refine_cv_ablations import (  # noqa: E402
    RefineCVNoDecisiveScenario,
    RefineCVQwenNoDecisiveScenario,
    RefineCVQwenCheckOnlyScenario,
    RefineCVQwenGapOnlyScenario,
    RefineCVQwenGapRefuteScenario,
)
from eid.scenarios.medclarify import MedClarifyScenario, MedClarifyMinQScenario  # noqa: E402
from eid.scenarios.mediq_abstention import MediQAbstentionScenario, MediQAbstentionMinQScenario  # noqa: E402
from eid.scenarios.uot import UoTScenario  # noqa: E402

PAPER_MODES = ["cot", "roleplay", "react", "sc", "refine"]

# 外部论文方法（零训练，移植到本场景当 baseline）：
#   medclarify      -> MedClarify (DEIG 信息增益选问题 + Bayesian 更新, arXiv:2602.17308)
#   mediq_abstention-> MEDIQ abstention (Scale+RG+Self-Consistency, NeurIPS'24)
#   uot             -> Uncertainty of Thoughts (信息增益模拟, arXiv:2402.03271)
#   medclarify_minq / mediq_minq -> 强制交互对照（至少问 4 轮，绕过 qwen 过早自信停止）
EXTERNAL_MODES = ["medclarify", "mediq_abstention", "uot",
                  "medclarify_minq", "mediq_minq"]

# 我们改过的方法（refine_cv 系列），在 16 turns / 200 例下重标定。
REFINE_CV_MODES = {
    "refine_cv": RefineCVScenario,                    # 跨模型 DeepSeek，四检查
    "refine_cv_qwen": RefineCVQwenScenario,           # 同模型 qwen，四检查
    "refine_cv_no_decisive": RefineCVNoDecisiveScenario,      # 跨模型，三检查(去 decisive)
    "refine_cv_qwen_no_decisive": RefineCVQwenNoDecisiveScenario,  # 同模型，三检查(去 decisive)
    "refine_cv_qwen_check": RefineCVQwenCheckOnlyScenario,   # 只 claim check
    "refine_cv_qwen_gap": RefineCVQwenGapOnlyScenario,       # 只 gap check
    "refine_cv_qwen_gap_refute": RefineCVQwenGapRefuteScenario,  # gap+refute(无 claim)
    # CEV-GR 移植到其它底座（sc 底座==refine_cv，不重复；react/roleplay 是真正的新底座）
    "react_cv": ReactCVScenario,        # ReAct 底座 + 四检查（通用移植，掉点）
    "react_cv2": ReactCV2Scenario,      # ReAct 底座 + 专用 CEV-GR（method-agnostic 门控）
    "roleplay_cv": RoleplayCVScenario,  # base 底座 + 四检查
}

ALL_MODES = PAPER_MODES + list(REFINE_CV_MODES.keys()) + EXTERNAL_MODES


def make_scenario(mode: str, cfg: "ModelConfig", max_turns: int):
    """按 mode 构造场景实例。

    - cot：静态单次，无 max_turns 概念。
    - refine：必须用上游原样 RefineCleanScenario（`== "INCOMPLETE"` 才重采）。
      老仓库的 RefineScenario 被团队改成 `in ("INCOMPLETE","REJECT")`，不是论文原样。
    - 其余 roleplay/react/sc 经查无团队改动，走 get_scenario 工厂即可。
    """
    if mode == "cot":
        return CoTScenario(dataset_name="medqa", doctor_config=cfg)
    if mode == "refine":
        return RefineCleanScenario(
            dataset_name="medqa",
            doctor_config=cfg,
            patient_config=cfg,
            reporter_config=cfg,
            max_turns=max_turns,
            summarizer_config=cfg,
            diagnostician_config=cfg,
            verifier_config=cfg,
        )
    if mode in REFINE_CV_MODES:
        cls = REFINE_CV_MODES[mode]
        return cls(
            dataset_name="medqa",
            doctor_config=cfg,
            patient_config=cfg,
            reporter_config=cfg,
            max_turns=max_turns,
            summarizer_config=cfg,
            diagnostician_config=cfg,
            verifier_config=cfg,
        )
    # 外部论文方法（零训练移植 baseline）
    # MedClarify 的 DEIG 每轮约 18 次 LLM 调用，qwen-plus 上置信度门控会"二元极端"
    # （50% turns=0、35% 问满 16 轮），问满 16 轮的单例约 48 分钟。加单例时间预算
    # 只做工程保护（不改方法逻辑），把"没收敛"记为 time_budget 截断。
    MC_TIME_BUDGET = float(os.environ.get("MC_TIME_BUDGET", "300"))
    if mode == "medclarify":
        return MedClarifyScenario(
            dataset_name="medqa", doctor_config=cfg, patient_config=cfg,
            reporter_config=cfg, max_turns=max_turns, time_budget=MC_TIME_BUDGET)
    if mode == "medclarify_minq":
        return MedClarifyMinQScenario(
            dataset_name="medqa", doctor_config=cfg, patient_config=cfg,
            reporter_config=cfg, max_turns=max_turns, time_budget=MC_TIME_BUDGET)
    if mode == "mediq_abstention":
        return MediQAbstentionScenario(
            dataset_name="medqa", doctor_config=cfg, patient_config=cfg,
            reporter_config=cfg, max_turns=max_turns)
    if mode == "mediq_minq":
        return MediQAbstentionMinQScenario(
            dataset_name="medqa", doctor_config=cfg, patient_config=cfg,
            reporter_config=cfg, max_turns=max_turns)
    if mode == "uot":
        return UoTScenario(
            dataset_name="medqa", doctor_config=cfg, patient_config=cfg,
            reporter_config=cfg, max_turns=max_turns)
    return get_scenario(
        mode=mode,
        dataset_name="medqa",
        doctor_config=cfg,
        patient_config=cfg,
        reporter_config=cfg,
        max_turns=max_turns,
        summarizer_config=cfg,
        diagnostician_config=cfg,
        verifier_config=cfg,
    )


def run_one(mode, cfg, max_turns, item, out_dir):
    """跑单个 case，返回 (status, case_id)。断点续跑：已存在非空则 skip。"""
    fn = out_dir / f"{item.case_id}.json"
    if fn.exists() and fn.stat().st_size > 0:
        return "SKIP", item.case_id

    scenario = make_scenario(mode, cfg, max_turns)
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
        fn.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        return "OK", item.case_id
    except Exception as e:  # noqa: BLE001
        # 不写 error 文件，允许后续重跑；错误信息打到 stderr
        print(f"[ERROR] {item.case_id}: {e}", file=sys.stderr, flush=True)
        return "ERR", item.case_id


def main():
    if len(sys.argv) < 2 or (sys.argv[1] not in ALL_MODES and sys.argv[1] not in ("all", "cv", "ext")):
        print(f"用法: python3 {sys.argv[0]} <mode> [N=200] [max_turns=16] [workers=10]")
        print(f"  mode: 论文5模式({', '.join(PAPER_MODES)}) | refine_cv系列({', '.join(REFINE_CV_MODES)})")
        print(f"        | 外部方法({', '.join(EXTERNAL_MODES)}) | all | cv | ext")
        sys.exit(2)

    mode_arg = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    max_turns = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    if mode_arg == "all":
        modes = ALL_MODES
    elif mode_arg == "cv":
        modes = list(REFINE_CV_MODES.keys())
    elif mode_arg == "ext":
        modes = EXTERNAL_MODES
    else:
        modes = [mode_arg]

    dataset = load_dataset("medqa")
    items = list(dataset)[:n]

    # 分片支持（拆多进程并行跑剩余 case）：SHARD_TOTAL 片里只跑 SHARD_ID 那一片。
    # 断点续跑机制（run_one 里 skip 已存在文件）保证分片间不重复、不冲突。
    shard_id = int(os.environ.get("SHARD_ID", "0"))
    shard_total = int(os.environ.get("SHARD_TOTAL", "1"))
    if shard_total > 1:
        items = [it for i, it in enumerate(items) if i % shard_total == shard_id]

    cfg = ModelConfig.from_string("qwen-plus")

    for mode in modes:
        out_dir = (
            EID_ROOT / "results" / "medqa" / mode
            / "qwen-plus" / f"{max_turns}_turns" / "record"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # 先统计已完成的（断点续跑）
        done = sum(
            1 for it in items
            if (out_dir / f"{it.case_id}.json").exists()
            and (out_dir / f"{it.case_id}.json").stat().st_size > 0
        )

        print(f"=== mode={mode}  n={len(items)}  max_turns={max_turns}  workers={workers}  (已存在 {done}) ===", flush=True)
        t_start = time.time()
        ok = skip = err = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(run_one, mode, cfg, max_turns, item, out_dir): item
                for item in items
            }
            for i, fut in enumerate(as_completed(futures), 1):
                status, cid = fut.result()
                if status == "OK":
                    ok += 1
                elif status == "SKIP":
                    skip += 1
                else:
                    err += 1
                if i % 10 == 0 or status == "ERR":
                    print(f"[{mode}] {i}/{len(items)}  ok={ok} skip={skip} err={err}  ({time.time()-t_start:.0f}s)", flush=True)

        print(f"=== {mode} 完成：ok={ok} skip={skip} err={err}  耗时 {(time.time()-t_start)/60:.1f} 分钟 ===", flush=True)


if __name__ == "__main__":
    main()

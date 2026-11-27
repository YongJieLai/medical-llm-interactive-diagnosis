"""Oracle 上界实验：分离「采集上限」与「诊断上限」。

背景（回答核心问题）：
  我们所有负结果（P0/两阶段/方案B/方案C1）都在 qwen-plus 上，此前提出"换更强模型"。
  但换模型之前必须先排除一个更根本的约束——信息上界：
    如果"有限 turns 下能采到的最优证据集合"本身就不足以让诊断器答对，
    那换什么模型、多聪明的采集器，都不可能选得更好。

本实验用同一个诊断器（qwen-plus，与 refine_cv_cover 完全一致）测诊断器的
纯判别上界（diagnostic ceiling），唯一变量是喂给它的证据：

  oracle_full : 完整 atomic_facts（病史 + 检查 全部）      —— 诊断器绝对上限
  oracle_key  : 仅 key_facts.json 标注的关键事实           —— "优先采关键证据"可达到的最优集合
  oracle_exam : 仅全部检查结果（exam_facts）               —— 检查结果本身是否足以确诊

若 oracle_key 就能到 ~90%+，则诊断器不是瓶颈，主战场=关键证据定向采集（不是换模型）。
若 oracle_full 也上不去，则诊断器判别能力就是瓶颈，才轮到"换诊断模型/引入外部知识"。

用法（多进程分片，每 case 独立文件、断点续跑）：
  for i in 0 1 2 3 4 5; do SHARD_ID=$i SHARD_TOTAL=6 python3 src/run_oracle_ceiling.py & done; wait

输出：
  results/oracle_ceiling/<case_id>.json   # 每 case 一个文件
"""
import os
import re
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))

from llm_client import get_llm, chat  # noqa: E402
from deterministic_judge import judge, get_deepseek  # noqa: E402
from eid.prompts.manager import PromptManager  # noqa: E402
from eid.scenarios.base import BaseScenario  # noqa: E402

DATA = REPO / "benchmarks" / "eid-benchmark" / "data" / "agentclinic_medqa_segmented.jsonl"
KEY_FACTS = REPO / "results" / "coverage_analysis" / "key_facts.json"
TRUTH_SRC = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa" / "refine_cv_cover" / "qwen-plus" / "16_turns" / "record"
OUT = REPO / "results" / "oracle_ceiling"
OUT.mkdir(parents=True, exist_ok=True)

PM = PromptManager(dataset_name="medqa")
DIAG_SYS = PM.get_diagnostician_cv_system_prompt()


def load_data() -> dict:
    """case_id -> dict(patient_facts, exam_facts)"""
    out = {}
    for line in DATA.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        af = d.get("atomic_facts", {}) or {}
        out[str(d.get("id"))] = {
            "patient_facts": af.get("patient_facts", []) or [],
            "exam_facts": af.get("exam_facts", []) or [],
        }
    return out


def load_truth() -> dict:
    """case_id -> ground_truth（与判分口径一致，取自 refine_cv_cover record）"""
    out = {}
    if TRUTH_SRC.exists():
        for f in TRUTH_SRC.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            out[d.get("case_id", f.stem)] = d.get("ground_truth", "")
    return out


def load_key_facts() -> dict:
    if KEY_FACTS.exists():
        return json.loads(KEY_FACTS.read_text())
    return {}


def pick_facts(facts: list, key_ids: set) -> list:
    out = []
    for f in facts:
        m = re.match(r"\s*(\d+)\.", f)
        if m and m.group(1) in key_ids:
            out.append(f)
    return out


def build_summary(patient_facts: list, exam_facts: list) -> str:
    parts = []
    if patient_facts:
        parts.append("Patient History:\n" + "\n".join(f"- {f}" for f in patient_facts))
    if exam_facts:
        parts.append("Physical Exam / Test Results:\n" + "\n".join(f"- {f}" for f in exam_facts))
    return "\n\n".join(parts)


def diagnose(client, model, summary: str) -> str:
    messages = [
        {"role": "system", "content": DIAG_SYS},
        {"role": "user", "content": f"### Case Summary ###\n{summary}\n\nDiagnostician:"},
    ]
    try:
        text = chat(client, model, messages, temperature=0.0, max_tokens=2048)
    except Exception as e:
        return f"__ERROR__: {e}"
    action, content = BaseScenario.extract_action(text)
    if action != "diagnosis":
        # 没有 [DIAGNOSIS] 段，退回整段文本
        return (text or "").strip()[:400]
    return content.strip()


def run_case(case_id: str) -> dict:
    data = load_data_global.get(case_id, {})
    truth = truth_global.get(case_id, "")
    kf = key_global.get(case_id, {})
    pf = data.get("patient_facts", [])
    ef = data.get("exam_facts", [])
    key_p = set(kf.get("key_patient", []) or [])
    key_e = set(kf.get("key_exam", []) or [])

    # 三层 oracle
    variants = {
        "oracle_full": (pf, ef),
        "oracle_key": (pick_facts(pf, key_p), pick_facts(ef, key_e)),
        "oracle_exam": ([], ef),
    }

    result = {
        "case_id": case_id,
        "truth": truth,
        "variants": {},
    }
    for name, (p, e) in variants.items():
        summary = build_summary(p, e)
        pred = diagnose(qwen_client, qwen_model, summary)
        verdict = judge(pred, truth, ds_client, ds_model) if truth else {"correct": None, "source": "no_truth"}
        result["variants"][name] = {
            "pred": pred,
            "correct": verdict.get("correct"),
            "source": verdict.get("source"),
            "n_patient": len(p),
            "n_exam": len(e),
        }
    return result


# 进程级全局（子进程 fork 后各自初始化）
load_data_global = {}
truth_global = {}
key_global = {}
qwen_client = None
qwen_model = None
ds_client = None
ds_model = None


def init_worker():
    global load_data_global, truth_global, key_global
    global qwen_client, qwen_model, ds_client, ds_model
    load_data_global = load_data()
    truth_global = load_truth()
    key_global = load_key_facts()
    qwen_client, qwen_model = get_llm("qwen")
    ds_client, ds_model = get_deepseek()


def main():
    shard_id = int(os.getenv("SHARD_ID", "0"))
    shard_total = int(os.getenv("SHARD_TOTAL", "1"))

    # 主进程先加载数据拿 case 列表（子进程各自再加载，这里只为分片）
    all_cases = sorted(load_data().keys())
    my_cases = all_cases[shard_id::shard_total]
    # 跳过已完成的 case
    todo = []
    for c in my_cases:
        p = OUT / f"{c}.json"
        if p.exists():
            try:
                j = json.loads(p.read_text())
                if "variants" in j and len(j["variants"]) >= 3:
                    continue
            except Exception:
                pass
        todo.append(c)

    print(f"[shard {shard_id}/{shard_total}] 总={len(my_cases)} 待跑={len(todo)}", flush=True)

    init_worker()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for result in ex.map(run_case, todo):
            (OUT / f"{result['case_id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            done += 1
            if done % 10 == 0:
                print(f"[shard {shard_id}] {done}/{len(todo)}", flush=True)

    print(f"[shard {shard_id}] DONE: {done} cases", flush=True)


if __name__ == "__main__":
    main()

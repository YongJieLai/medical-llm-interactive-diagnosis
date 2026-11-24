"""对比所有变体的确定性准确率 + dump 规则不确定 case 供人工复核。

口径：rule_judge 返回 True = 对；False/None = 错（None 单独 dump 供复核，
因为新变体会产生新 prediction，不能直接假设 None 全错）。

方法 -> record 目录映射（与 run_variants.py 输出路径一致）。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge, normalize

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")

METHODS = {
    "refine": "refine/qwen-plus/10_turns",
    "refine_cv(跨模型DeepSeek)": "refine_cv/qwen-plus/10_turns",
    "refine_cv_qwen(同模型)": "refine_cv_qwen/qwen-plus/10_turns",
    "check_only(只核对)": "refine_cv_qwen_check/qwen-plus/10_turns",
    "gap_only(只缺口)": "refine_cv_qwen_gap/qwen-plus/10_turns",
    "refute_only(只反证)": "refine_cv_qwen_refute/qwen-plus/10_turns",
    "qwenmax(强跨模型)": "refine_cv_qwenmax/qwen-plus/10_turns",
    "all_deepseek(同模型DS)": "refine_cv/deepseek/10_turns",
}


def load(method_dir: Path) -> dict:
    out = {}
    record_dir = method_dir / "record"
    if not record_dir.is_dir():
        return out
    for fn in sorted(record_dir.iterdir()):
        if fn.suffix != ".json":
            continue
        try:
            r = json.loads(fn.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "case_id" not in r:
            continue
        out[r["case_id"]] = {
            "pred": (r.get("prediction") or "").strip(),
            "truth": (r.get("ground_truth") or "").strip(),
        }
    return out


def main():
    data = {}
    for label, rel in METHODS.items():
        rows = load(BASE / rel)
        data[label] = rows
        if rows:
            print(f"[读入] {label}: {len(rows)} 例", flush=True)
        else:
            print(f"[缺失] {label}: {rel} 尚无 record", flush=True)

    print()
    print(f"{'方法':28s} {'正确':>4s} {'总数':>4s} {'准确率':>7s} {'不确定':>4s}")
    print("-" * 60)
    uncertain_all = {}
    for label, rows in data.items():
        if not rows:
            continue
        true = false = none = 0
        for cid, v in rows.items():
            r = rule_judge(v["pred"], v["truth"])
            if r is True:
                true += 1
            elif r is False:
                false += 1
            else:
                none += 1
                uncertain_all.setdefault(cid, {})[label] = (v["truth"], v["pred"])
        n = len(rows)
        # 口径：True=对；False/None 暂按错（None 待复核）
        acc = true / n * 100
        print(f"{label:28s} {true:>4d} {n:>4d} {acc:>6.1f}% {none:>4d}")

    print()
    print("=== 规则不确定 case（需人工复核，按 case_id 汇总）===")
    for cid in sorted(uncertain_all):
        for label, (truth, pred) in uncertain_all[cid].items():
            print(f"[{cid}] truth={truth[:45]!r}")
            print(f"     [{label}] pred={pred[:65]!r}")

    out = {
        "uncertain": {
            cid: {lab: {"truth": t, "pred": p} for lab, (t, p) in v.items()}
            for cid, v in uncertain_all.items()
        }
    }
    Path("/workspace/medical_llm_exp/results/variants_uncertain.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    print()
    print("已落盘 results/variants_uncertain.json")


if __name__ == "__main__":
    main()

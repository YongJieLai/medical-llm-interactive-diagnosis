"""对比 refute 系列消融：gap_only vs gap_refute vs gap_refute_rag。

目标：测出 REFUTATION 的真实净贡献（修复死锁后），以及 DDX-KB 知识注入的额外贡献。

口径：rule_judge 确定性判定（True=对；False/None 暂按错，None 单独 dump 供人工复核）。

refute 触发检测：trace 里 doctor 收到含 "unchanged from the previous round" 的
feedback（即 anchored GATHER）的次数。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")

METHODS = {
    "gap_only(只缺口)": "refine_cv_qwen_gap/qwen-plus/10_turns",
    "gap_refute(缺口+反证)": "refine_cv_qwen_gap_refute/qwen-plus/10_turns",
    "gap_refute_rag(缺口+反证+KB)": "refine_cv_qwen_gap_refute_rag/qwen-plus/10_turns",
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
        out[r["case_id"]] = r
    return out


def refute_stats(record: dict) -> dict:
    """统计单个 record 里 refute 的触发情况。"""
    trace = record.get("trace", [])
    verifier_calls = 0
    refute_triggers = 0
    rag_hits = 0
    for t in trace:
        rid = t.get("role_id")
        if rid == "verifier":
            verifier_calls += 1
            c = t.get("content")
            if isinstance(c, dict):
                ref = c.get("refutation", {}) or {}
                if ref.get("counter_evidence") or ref.get("alternative"):
                    rag_hits += 1  # verifier 输出了反证/备择（含 KB 驱动）
        elif rid == "doctor":
            c = t.get("content", "")
            if "unchanged from the previous round" in str(c):
                refute_triggers += 1
    return {
        "verifier_calls": verifier_calls,
        "refute_triggers": refute_triggers,
        "verifier_with_refutation": rag_hits,
    }


def main():
    data = {}
    for label, rel in METHODS.items():
        rows = load(BASE / rel)
        data[label] = rows
        print(f"[读入] {label}: {len(rows)} 例", flush=True)

    # 用 gap_only 的 case 集合作为对齐基准
    base_cases = set(data["gap_only(只缺口)"].keys())
    print(f"\n对齐基准 case 数: {len(base_cases)}\n")

    print(f"{'方法':30s} {'正确':>4s} {'总数':>4s} {'准确率':>7s} {'不确定':>4s} {'refute触发':>6s} {'avg.verif':>8s}")
    print("-" * 80)
    uncertain_all = {}
    trigger_all = {}
    for label, rows in data.items():
        if not rows:
            continue
        true = false = none = 0
        trig_sum = 0
        verif_sum = 0
        trig_cases = set()
        for cid, r in rows.items():
            if cid not in base_cases:
                continue
            pred = (r.get("prediction") or "").strip()
            truth = (r.get("ground_truth") or "").strip()
            res = rule_judge(pred, truth)
            if res is True:
                true += 1
            elif res is False:
                false += 1
            else:
                none += 1
                uncertain_all.setdefault(cid, {})[label] = (truth, pred)
            st = refute_stats(r)
            trig_sum += st["refute_triggers"]
            verif_sum += st["verifier_calls"]
            if st["refute_triggers"] > 0:
                trig_cases.add(cid)
        n = len(base_cases)
        acc = true / n * 100
        trigger_all[label] = trig_cases
        avg_verif = verif_sum / n if n else 0
        print(f"{label:30s} {true:>4d} {n:>4d} {acc:>6.1f}% {none:>4d} {len(trig_cases):>6d} {avg_verif:>8.2f}")

    print("\n=== refute 触发 case 分布（gap_refute vs gap_refute_rag）===")
    if "gap_refute(缺口+反证)" in trigger_all and "gap_refute_rag(缺口+反证+KB)" in trigger_all:
        a = trigger_all["gap_refute(缺口+反证)"]
        b = trigger_all["gap_refute_rag(缺口+反证+KB)"]
        print(f"  gap_refute 触发 {len(a)} 例: {sorted(a)}")
        print(f"  gap_refute_rag 触发 {len(b)} 例: {sorted(b)}")
        print(f"  两者都触发: {len(a & b)} 例")

    print("\n=== 规则不确定 case（需人工复核）===")
    for cid in sorted(uncertain_all):
        for label, (truth, pred) in uncertain_all[cid].items():
            print(f"[{cid}] truth={truth[:50]!r}")
            print(f"     [{label}] pred={pred[:70]!r}")

    out = {
        "uncertain": {
            cid: {lab: {"truth": t, "pred": p} for lab, (t, p) in v.items()}
            for cid, v in uncertain_all.items()
        }
    }
    Path("/workspace/medical_llm_exp/results/refute_uncertain.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    print("\n已落盘 results/refute_uncertain.json")


if __name__ == "__main__":
    main()

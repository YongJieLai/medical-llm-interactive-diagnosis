"""对比 meddxagent（基线）vs meddxagent_gate（方法无关闸门插件）的确定性准确率。

口径：rule_judge 规则层（纯确定性）。规则不确定(None)的 case 单独 dump 供人工复核。
与 CEV-GR 消融报告的判定口径一致：不用 DeepSeek 兜底 judge（已证有噪声）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge, normalize

BASE = Path("/workspace/medical_llm_exp/benchmarks/eid-benchmark/results/medqa")

METHODS = {
    "meddxagent(基线)": "meddxagent/qwen-plus/10_turns",
    "meddxagent_gate(插件)": "meddxagent_gate/qwen-plus/10_turns",
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
        print(f"[读入] {label}: {len(rows)} 例", flush=True)

    print()
    print(f"{'方法':24s} {'正确':>4s} {'总数':>4s} {'准确率':>7s} {'不确定':>4s}")
    print("-" * 56)

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
        acc = true / n * 100
        print(f"{label:24s} {true:>4d} {n:>4d} {acc:>6.1f}% {none:>4d}")

    # 逐 case 对齐对比（规则层口径：True=对，False=错，None=不确定）
    print()
    print("=== 逐 case 对齐（baseline vs gate，规则层口径）===")
    print(f"{'case_id':22s} {'基线':>6s} {'插件':>6s}  真值")
    baselines = data.get("meddxagent(基线)", {})
    gates = data.get("meddxagent_gate(插件)", {})
    for cid in sorted(set(baselines) | set(gates)):
        b = baselines.get(cid)
        g = gates.get(cid)
        truth = (b or g or {}).get("truth", "")
        rb = rule_judge(b["pred"], truth) if b else None
        rg = rule_judge(g["pred"], truth) if g else None
        def sym(x):
            return "✓" if x is True else ("✗" if x is False else "?")
        print(f"{cid:22s} {sym(rb):>6s} {sym(rg):>6s}  {truth[:45]}")

    print()
    print("=== 规则不确定 case（需人工复核）===")
    for cid in sorted(uncertain_all):
        for label, (truth, pred) in uncertain_all[cid].items():
            print(f"[{cid}] truth={truth!r}")
            print(f"     [{label}] pred={pred!r}")

    out = {"uncertain": {}}
    for cid, v in uncertain_all.items():
        out["uncertain"][cid] = {lab: {"truth": t, "pred": p} for lab, (t, p) in v.items()}
    Path("/workspace/medical_llm_exp/results/gate_uncertain.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    print()
    print("已落盘 results/gate_uncertain.json")


if __name__ == "__main__":
    main()

"""P0: 规则层重判 GapRefute vs GapOnly，测「反证的净贡献」（零 API）。

背景：旧 RefuteOnly 消融因结构性死锁（反证从未触发）作废。正确测法是用
GapRefute（gap+refute）对比 GapOnly（只有 gap），差值 = 反证的净贡献。

本脚本只做规则层（rule_judge，零 API）：
  - 规则层确定 CORRECT / INCORRECT 的记入；
  - 规则层判不出的（None）记入「不确定」，需后续 LLM 兜底或人工复核。
口径：规则层准确率 = CORRECT / 总数（「不确定」暂按错，与 HANDOFF.md §4 一致）。

运行：python3 src/p0_refute_net_contribution.py
"""
import json
import sys
import types
from collections import Counter
from pathlib import Path

# Stub llm_client so deterministic_judge imports without the openai dependency
# (rule_judge / normalize use only the stdlib; the LLM fallback is unused here).
_llm_client = types.ModuleType("llm_client")
_llm_client.get_llm = lambda *a, **k: (None, None)
_llm_client.chat = lambda *a, **k: None
sys.modules["llm_client"] = _llm_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_judge import rule_judge, normalize  # noqa: E402

BASE = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "eid-benchmark" / "results" / "medqa"
)


def load_records(method: str, n: int):
    rec_dir = BASE / method / "qwen-plus" / "10_turns" / "record"
    out = []
    for i in range(n):
        fn = rec_dir / f"agentclinic_medqa_{i}.json"
        if not fn.exists():
            print(f"  [缺] {fn.name}")
            continue
        r = json.loads(fn.read_text())
        out.append(r)
    return out


def judge_method(method: str, n: int):
    recs = load_records(method, n)
    stats = Counter()
    uncertain = []
    for r in recs:
        pred = r.get("prediction") or ""
        truth = r.get("ground_truth") or ""
        verdict = rule_judge(pred, truth)
        if verdict is True:
            stats["correct"] += 1
        elif verdict is False:
            stats["incorrect"] += 1
        else:
            stats["uncertain"] += 1
            uncertain.append({
                "case": r.get("case_id", "?"),
                "pred": pred,
                "truth": truth,
            })
    return stats, uncertain


def main():
    # GapRefute 只有 30 例；GapOnly 有 60 例，取前 30 例对齐（id 0-29）
    n = 30
    print(f"规则层重判（前 {n} 例，id 0-{n-1}）\n")

    methods = ["refine_cv_qwen_gap_refute", "refine_cv_qwen_gap"]
    results = {}
    for m in methods:
        stats, uncertain = judge_method(m, n)
        total = stats["correct"] + stats["incorrect"] + stats["uncertain"]
        acc = stats["correct"] / total * 100 if total else 0
        results[m] = (stats, acc, uncertain)
        print(f"[{m}]")
        print(f"  correct={stats['correct']}  incorrect={stats['incorrect']}  "
              f"uncertain={stats['uncertain']}  n={total}")
        print(f"  规则层准确率（不确定按错）= {acc:.1f}%")
        print()

    # 反证净贡献 = GapRefute - GapOnly
    gaprefute_acc = results["refine_cv_qwen_gap_refute"][1]
    gaponly_acc = results["refine_cv_qwen_gap"][1]
    net = gaprefute_acc - gaponly_acc
    print("=" * 60)
    print(f"反证净贡献（GapRefute - GapOnly）= {net:+.1f} pp")
    print("=" * 60)

    # dump 不确定清单（供后续 LLM 兜底 / 人工复核）
    out = {m: results[m][2] for m in methods}
    outp = Path(__file__).resolve().parent.parent / "results" / "p0_refute_uncertain.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n不确定清单已写入 {outp}")


if __name__ == "__main__":
    main()

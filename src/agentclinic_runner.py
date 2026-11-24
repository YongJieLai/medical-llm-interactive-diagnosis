"""AgentClinic 批量运行器：多病例 × 方法，采集结构化 trace 到 jsonl。

用法：
  python3 -m src.agentclinic_runner --split medqa --providers qwen --limit 20 --verbose
"""
import json
import time
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_llm
from src.agentclinic_env import load_medqa_cases, run_vanilla, is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="medqa", help="medqa/medqa_extended/nejm/nejm_extended")
    ap.add_argument("--providers", default="qwen")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results" / "agentclinic_runs.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = load_medqa_cases(args.split, limit=args.limit)
    client, model = get_llm(args.providers)
    print(f"provider={args.providers} model={model} split={args.split} cases={len(cases)}")
    print(f"输出: {args.out}")
    print("-" * 60)

    done = correct = 0
    with open(args.out, "a") as f:
        for case in cases:
            t0 = time.time()
            r = run_vanilla(client, model, case, max_turns=args.max_turns, verbose=args.verbose)
            r["correct"] = is_correct(r["diagnosis"], r["truth"])
            r["elapsed_sec"] = round(time.time() - t0, 1)
            r["provider"] = args.providers
            r["model"] = model
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            correct += int(r["correct"])
            print(f"{'✓' if r['correct'] else '✗'} {case['case_id']} "
                  f"-> {r['diagnosis']!r} (truth={r['truth']!r}, {r['turns']}t, {r['elapsed_sec']}s)")

    print("-" * 60)
    print(f"完成 {done}，正确 {correct}/{done}（{correct / done:.1%}）")


if __name__ == "__main__":
    main()

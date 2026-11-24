"""实验运行器：方法 × 模型 × 病例 全组合，采集完整过程。

用法：
  python3 -m src.runner --methods vanilla,medkgi,sgur,knowguard \
      --providers deepseek --limit 3

结果逐条追加到 results/runs.jsonl（可断点续跑）。
"""
import json
import time
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ddxplus_loader import load_evidences, load_conditions
from src.patient import ScriptPatient
from src.llm_client import get_llm, available_providers
from src.methods import METHODS


def normalize(s):
    return (s or "").strip().lower().replace(" ", "").replace("_", "")


def is_correct(diag, pathology):
    """诊断判对：精确匹配或规范化后互相包含。"""
    if not diag:
        return False
    if normalize(diag) == normalize(pathology):
        return True
    # 处理 "Possible NSTEMI / STEMI" vs "NSTEMI" 这类
    d, p = normalize(diag), normalize(pathology)
    return (d in p) or (p in d)


def run_one(method_name, provider, case, ev, cond, verbose=False):
    client, model = get_llm(provider)
    method = METHODS[method_name](client, model, ev, cond)
    patient = ScriptPatient(case, ev)
    t0 = time.time()
    result = method.run(patient)
    result["correct"] = is_correct(result["diagnosis"], case["pathology"])
    result.update({
        "provider": provider,
        "model": model,
        "case_id": case["case_id"],
        "pathology": case["pathology"],
        "elapsed_sec": round(time.time() - t0, 1),
        "n_gt_evidence": len(case["evidence_values"]),
    })
    if verbose:
        status = "✓" if result["correct"] else "✗"
        print(f"  {status} {method_name}/{provider} {case['case_id']} "
              f"-> {result['diagnosis']!r} (truth={case['pathology']!r}, {result['turns']} turns)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="vanilla,medkgi,sgur,knowguard")
    ap.add_argument("--providers", default="", help="逗号分隔，默认用所有已配置的")
    ap.add_argument("--limit", type=int, default=0, help="最多跑几个病例")
    ap.add_argument("--case-ids", default="", help="逗号分隔，指定病例")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ev = load_evidences()
    cond = load_conditions()
    with open(ROOT / "results" / "cases.json") as f:
        cases = json.load(f)["cases"]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()] or available_providers()
    if not providers:
        print("没有可用的 provider，请先设置 API key 环境变量（DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / ZHIPU_API_KEY）")
        sys.exit(1)

    if args.case_ids:
        want = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    if args.limit:
        cases = cases[: args.limit]

    print(f"方法: {methods}")
    print(f"模型: {providers}")
    print(f"病例: {len(cases)} 个")
    print(f"组合数: {len(methods) * len(providers) * len(cases)}")
    print(f"输出: {args.out}")
    print("-" * 60)

    done = 0
    with open(args.out, "a") as fout:
        for mi, m in enumerate(methods):
            for pi, p in enumerate(providers):
                for ci, case in enumerate(cases):
                    try:
                        r = run_one(m, p, case, ev, cond, verbose=args.verbose)
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fout.flush()
                        done += 1
                    except Exception as e:
                        print(f"[FAIL] {m}/{p} {case['case_id']}: {type(e).__name__}: {e}")
                        fout.write(json.dumps({
                            "method": m, "provider": p, "case_id": case["case_id"],
                            "pathology": case["pathology"], "error": str(e),
                        }, ensure_ascii=False) + "\n")
                        fout.flush()
    print("-" * 60)
    print(f"完成 {done} 条，结果在 {args.out}")


if __name__ == "__main__":
    main()

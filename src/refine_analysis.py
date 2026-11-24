"""REFINE (EID-Benchmark) 批量结果的失效分析脚本。

读取 results/<dataset>/<mode>/<model>/<turns>/record/*.json，
汇总诊断对错、信息覆盖率，并对失败病例抽取可读的失效线索。
"""
import json
import os
import sys
from collections import Counter, defaultdict


def load_records(base_dir):
    rec_dir = os.path.join(base_dir, "record")
    records = []
    if not os.path.isdir(rec_dir):
        return records
    for fn in sorted(os.listdir(rec_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(rec_dir, fn)) as f:
                records.append(json.load(f))
    return records


def summarize(records, title):
    n = len(records)
    if n == 0:
        print(f"\n### {title}: 无数据\n")
        return
    correct = sum(1 for r in records if r.get("is_correct"))
    cov = [r.get("coverage") or {} for r in records]
    icr = [c.get("information_coverage_rate") for c in cov if c.get("information_coverage_rate") is not None]
    pcov = [c.get("patient_coverage") for c in cov if c.get("patient_coverage") is not None]
    ecov = [c.get("exam_coverage") for c in cov if c.get("exam_coverage") is not None]

    print(f"\n### {title}  (n={n})")
    print(f"  诊断正确率: {correct}/{n} = {correct/n*100:.1f}%")
    if icr:
        print(f"  信息覆盖率 ICR: 均值={sum(icr)/len(icr):.3f}")
    if pcov:
        print(f"  患者事实覆盖: 均值={sum(pcov)/len(pcov):.3f}")
    if ecov:
        print(f"  检查事实覆盖: 均值={sum(ecov)/len(ecov):.3f}")

    # 调用量
    calls = [len(r.get("trace", [])) for r in records]
    if calls:
        print(f"  每病例 LLM 调用: 均值={sum(calls)/len(calls):.1f}  min={min(calls)}  max={max(calls)}")

    return correct, n


def failure_cases(records, max_show=10):
    """打印失败病例的关键线索。"""
    fails = [r for r in records if not r.get("is_correct")]
    print(f"\n--- 失败病例 (共 {len(fails)} 例，显示前 {max_show}) ---")
    for r in fails[:max_show]:
        pred = r.get("prediction")
        truth = r.get("ground_truth")
        cov = r.get("coverage") or {}
        icr = cov.get("information_coverage_rate")
        pcov = cov.get("patient_coverage")
        ecov = cov.get("exam_coverage")
        trace = r.get("trace", [])
        turns = [t for t in trace if t.get("role_id") == "doctor"]
        n_turns = len(turns)
        has_verifier = any(t.get("role_id") in ("verifier", "diagnostician") for t in trace)
        print(f"  [病例 {r.get('case_id', '?')}] 真值={truth!r} 预测={pred!r}")
        print(f"      ICR={icr if icr is not None else 'N/A'} 患者覆盖={pcov if pcov is not None else 'N/A'} 检查覆盖={ecov if ecov is not None else 'N/A'}  医生轮数={n_turns}  有验证环节={has_verifier}")
    return fails


def doctor_dialogue(record, max_turns=20):
    """从 metadata.role_records 提取医生提问与患者回答，便于人读。"""
    md = record.get("metadata") or {}
    rr = md.get("role_records") or []
    lines = []
    for item in rr:
        if not isinstance(item, dict):
            continue
        for role, msgs in item.items():
            if role == "token_usage" or not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                content = m.get("content", "")
                if not content or not isinstance(content, str):
                    continue
                content = content.strip()
                if not content:
                    continue
                role_name = {"doctor": "医生", "patient": "患者", "reporter": "报告者",
                             "summarizer": "总结", "diagnostician": "诊断者",
                             "verifier": "验证者", "judge": "裁判"}.get(role, role)
                lines.append(f"[{role_name}] {content}")
    return lines


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = {}
    for dataset in os.listdir(base):
        dpath = os.path.join(base, dataset)
        if not os.path.isdir(dpath):
            continue
        for mode in os.listdir(dpath):
            mpath = os.path.join(dpath, mode)
            if not os.path.isdir(mpath):
                continue
            for model in os.listdir(mpath):
                modelpath = os.path.join(mpath, model)
                if not os.path.isdir(modelpath):
                    continue
                for turns in os.listdir(modelpath):
                    tpath = os.path.join(modelpath, turns)
                    if not os.path.isdir(tpath):
                        continue
                    records = load_records(tpath)
                    key = f"{dataset}/{mode}/{model}/{turns}"
                    res = summarize(records, key)
                    out[key] = res
                    if mode in ("refine", "react", "sc") and records:
                        failure_cases(records)

    print("\n\n=== 汇总表 ===")
    print(f"{'dataset/mode/model/turns':45s} {'对/总':>10s} {'准确率':>8s}")
    for k, v in out.items():
        if v and v[1]:
            print(f"{k:45s} {v[0]:>4d}/{v[1]:<4d} {v[0]/v[1]*100:>6.1f}%")


if __name__ == "__main__":
    main()

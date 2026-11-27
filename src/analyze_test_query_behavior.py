"""医生采集行为分析：query（问诊）vs test（检查）的时序模式。

回答：医生什么时候下检查单、什么时候转向询问，以及检查覆盖低的真实机制。

从 trace 提取每例 doctor 的 action 序列（[QUERY]/[TEST]/[FINISH]）+ reporter 的
synthetic 标记（下错检查名→合成中性结果），做：
  1. 检查/询问的总体比例
  2. 首次 test 的轮次分布（医生多久才开始查）
  3. test 在时间轴上的密度（前/中/后段）
  4. 转向模式（query→test 与 test→query 的转换点）
  5. test 有效性（synthetic 占比 = 下错检查名）
  6. 正确组 vs 错误组的 test 行为差异
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks" / "eid-benchmark" / "src"))
from eid.scenarios.base import BaseScenario  # noqa: E402

BASE = REPO / "benchmarks" / "eid-benchmark" / "results" / "medqa"
MODES = ["refine_cv", "refine_cv_cover", "refine_cv_discrim"]
PC = REPO / "results" / "coverage_analysis"


def load_records(mode):
    d = {}
    rec_dir = BASE / mode / "qwen-plus" / "16_turns" / "record"
    if not rec_dir.exists():
        return d
    for f in rec_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        d[r["case_id"]] = r
    return d


def load_correct(mode):
    p = PC / f"percase_{mode}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {c: v.get("correct") for c, v in d.items()}


def parse_actions(trace):
    """返回 [(turn_idx, action_type, is_test_synthetic)] 只取 doctor 的动作，
    synthetic 从紧随其后的 reporter 条目关联。"""
    actions = []
    reporter_synth = {}
    for e in trace:
        if e.get("role_id") == "reporter":
            # 记录该 reporter 条目是否 synthetic（供后续关联）
            pass
    # 逐条扫描：doctor 条目解析 action；reporter 条目的 synthetic 关联到上一个 doctor test
    last_test_synth = None
    for i, e in enumerate(trace):
        rid = e.get("role_id", "")
        content = e.get("content", "")
        if rid == "doctor":
            at, ac = BaseScenario.extract_action(content)
            if at in ("query", "test", "finish"):
                actions.append({"turn": len(actions), "type": at, "content": ac, "synthetic": None})
        elif rid == "reporter":
            synth = e.get("synthetic", False)
            # 关联到最近一个 test
            for a in reversed(actions):
                if a["type"] == "test" and a["synthetic"] is None:
                    a["synthetic"] = synth
                    break
    return actions


def analyze(mode):
    recs = load_records(mode)
    corr = load_correct(mode)
    stats = defaultdict(list)
    percase = []
    for cid, r in recs.items():
        actions = parse_actions(r["trace"])
        queries = [a for a in actions if a["type"] == "query"]
        tests = [a for a in actions if a["type"] == "test"]
        finishes = [a for a in actions if a["type"] == "finish"]
        n_q, n_t = len(queries), len(tests)
        total = n_q + n_t
        # 首次 test 的轮次（1-based）
        first_test = tests[0]["turn"] + 1 if tests else None
        # 转向次数
        q2t = sum(1 for i in range(1, len(actions))
                  if actions[i-1]["type"] == "query" and actions[i]["type"] == "test")
        t2q = sum(1 for i in range(1, len(actions))
                  if actions[i-1]["type"] == "test" and actions[i]["type"] == "query")
        # synthetic 测试占比
        synth_n = sum(1 for a in tests if a["synthetic"])
        # 前/中/后段 test 密度
        seg = {"early": 0, "mid": 0, "late": 0}
        for a in tests:
            pos = a["turn"] / max(total, 1)
            if pos < 0.33:
                seg["early"] += 1
            elif pos < 0.66:
                seg["mid"] += 1
            else:
                seg["late"] += 1
        percase.append({
            "cid": cid, "n_q": n_q, "n_t": n_t, "first_test": first_test,
            "q2t": q2t, "t2q": t2q, "synth": synth_n,
            "seg": seg, "correct": corr.get(cid),
        })
    return percase


def report(mode):
    pc = analyze(mode)
    n = len(pc)
    n_q = [c["n_q"] for c in pc]
    n_t = [c["n_t"] for c in pc]
    tot_q, tot_t = sum(n_q), sum(n_t)
    ft = [c["first_test"] for c in pc if c["first_test"] is not None]
    q2t = [c["q2t"] for c in pc]
    t2q = [c["t2q"] for c in pc]
    synth = [c["synth"] for c in pc]
    tot_synth = sum(synth)

    # 正确 vs 错误
    corr_pc = [c for c in pc if c["correct"] is True]
    inc_pc = [c for c in pc if c["correct"] is False]

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print(f"\n{'='*70}")
    print(f"模式 {mode}  (n={n})")
    print(f"{'='*70}")
    print(f"动作总量: query={tot_q}  test={tot_t}  →  检查占比 = {tot_t/(tot_q+tot_t):.1%}")
    print(f"每例平均: query={avg(n_q):.1f}  test={avg(n_t):.1f}")
    print(f"首次 test 轮次: 中位={sorted(ft)[len(ft)//2] if ft else 'NA'}  均值={avg(ft):.1f}  (16轮上限)")
    print(f"转向: query→test 每例 {avg(q2t):.1f} 次,  test→query 每例 {avg(t2q):.1f} 次")
    print(f"test 有效性: synthetic(下错检查名)={tot_synth}/{tot_t} = {tot_synth/tot_t:.1%}" if tot_t else "无test")
    # 分段 test 密度
    seg_tot = defaultdict(int)
    for c in pc:
        for k, v in c["seg"].items():
            seg_tot[k] += v
    print(f"test 时间分布: 前段={seg_tot['early']} 中段={seg_tot['mid']} 后段={seg_tot['late']}")
    # 首次 test 分布
    ft_dist = Counter(ft)
    print(f"首次 test 轮次分布: " + ", ".join(f"第{t}轮:{ft_dist.get(t,0)}" for t in sorted(ft_dist)))
    # 正确 vs 错误
    if corr_pc and inc_pc:
        print(f"正确组(n={len(corr_pc)}): 平均test={avg([c['n_t'] for c in corr_pc]):.1f}  首次test轮次={avg([c['first_test'] for c in corr_pc if c['first_test']]):.1f}")
        print(f"错误组(n={len(inc_pc)}): 平均test={avg([c['n_t'] for c in inc_pc]):.1f}  首次test轮次={avg([c['first_test'] for c in inc_pc if c['first_test']]):.1f}")
    return pc


if __name__ == "__main__":
    for m in MODES:
        report(m)

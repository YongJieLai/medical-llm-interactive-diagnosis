"""Offline unit tests for HypothesisTracker (no LLM calls, no camel dependency).

Loads hypothesis_tracker.py and refine_ht.py directly via importlib, bypassing
the eid package __init__ (which pulls in camel). Verifies the mechanics that
matter for the "correct hypothesis dropped" failure mode.

Run:  python3 src/test_hypothesis_tracker.py
"""
import importlib.util
import sys
import types
from pathlib import Path

# eid package lives at benchmarks/eid-benchmark/src/eid (relative to this file)
SRC = Path(__file__).resolve().parent.parent / "benchmarks" / "eid-benchmark" / "src"


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 1. load hypothesis_tracker (pure stdlib) ----
ht = load_module("eid.hypothesis_tracker", SRC / "eid" / "hypothesis_tracker.py")
HypothesisTracker = ht.HypothesisTracker
normalize_name = ht.normalize_name

# ---- 2. stub eid.scenarios.refine_cv_qwen, then load refine_ht ----
fake_refine_cv_qwen = types.ModuleType("eid.scenarios.refine_cv_qwen")
class FakeRefineCVQwenScenario:
    pass
fake_refine_cv_qwen.RefineCVQwenScenario = FakeRefineCVQwenScenario
sys.modules["eid.scenarios.refine_cv_qwen"] = fake_refine_cv_qwen

refine_ht = load_module("eid.scenarios.refine_ht", SRC / "eid" / "scenarios" / "refine_ht.py")


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


results = []

print("=== 1. normalize_name fuzzy dedup ===")
results.append(check("possessive 's", normalize_name("Hirschsprung's disease") == "hirschsprung disease"))
results.append(check("parenthetical alias stripped", normalize_name("Lung cancer (with SIADH)") == "lung cancer"))
results.append(check("case/punct", normalize_name("  Bowen's Disease!! ") == "bowen disease"))

print("\n=== 2. cross-round accumulation (survives re-summarisation) ===")
t = HypothesisTracker()
t.begin_round(); t.register("UTI", supporting=["dysuria"])
t.begin_round(); t.register("UTI", supporting=["pyuria"])
d = t.to_dict()
results.append(check("one candidate after 2 rounds", len(d["hypotheses"]) == 1))
results.append(check("rounds_seen = [1,2]", d["hypotheses"][0]["rounds_seen"] == [1, 2]))
results.append(check("evidence accumulated", d["hypotheses"][0]["supporting"] == ["dysuria", "pyuria"]))

print("\n=== 3. fuzzy dedup across rounds (name variants collapse) ===")
t2 = HypothesisTracker()
t2.begin_round(); t2.register("Hirschsprung's disease")
t2.begin_round(); t2.register("Hirschsprung disease")
results.append(check("variants collapse to 1", len(t2.to_dict()["hypotheses"]) == 1))
results.append(check("rounds_seen = [1,2] (no dup)", t2.to_dict()["hypotheses"][0]["rounds_seen"] == [1, 2]))

print("\n=== 4. supporting / refuting / excluded rendering ===")
t3 = HypothesisTracker()
t3.begin_round(); t3.register("Gout", supporting=["elevated urate"], refuting=["no podagra"])
t3.begin_round(); t3.register("Migraine", excluded=True, exclusion_reason="no aura, no photophobia")
r = t3.render()
results.append(check("active shown", "Gout" in r and "Still viable" in r))
results.append(check("excluded partitioned", "Migraine" in r and "ruled out" in r))
results.append(check("supporting rendered", "elevated urate" in r))
results.append(check("against rendered", "no podagra" in r))
results.append(check("exclusion reason rendered", "no aura" in r))
results.append(check("memory-aid framing (not constraint)", "NOT a constraint" in r))

print("\n=== 5. LRU eviction under overflow ===")
t4 = HypothesisTracker(max_candidates=2)
t4.begin_round(); t4.register("A")
t4.begin_round(); t4.register("B")
t4.begin_round(); t4.register("C")
names = sorted(h["name"] for h in t4.to_dict()["hypotheses"])
results.append(check("oldest (A) evicted", names == ["B", "C"]))
results.append(check("size capped at 2", len(t4.to_dict()["hypotheses"]) == 2))

print("\n=== 6. [EVIDENCE] extraction ===")
sample = (
    "[THOUGHT] ...\n"
    "[EVIDENCE]\n"
    "- unable to extend the distal interphalangeal joint\n"
    "- tenderness over the dorsal finger\n"
    "[GAPS]\n"
    "- whether there was trauma\n"
    "[DIAGNOSIS]\nMallet finger\n"
)
ev = refine_ht.RefineHTScenario._extract_evidence(sample)
results.append(check("extracted 2 statements", ev == [
    "unable to extend the distal interphalangeal joint",
    "tenderness over the dorsal finger",
]))
results.append(check("stops before [GAPS]", all("trauma" not in e for e in ev)))

print("\n=== 7. RefineHTScenario wiring sanity ===")
results.append(check("inherits RefineCVQwenScenario", issubclass(refine_ht.RefineHTScenario, FakeRefineCVQwenScenario)))

print("\n" + "=" * 50)
passed = sum(results)
print(f"结果: {passed}/{len(results)} 通过")
sys.exit(0 if passed == len(results) else 1)

"""离线评估 reporter 端归一化器（不调用 LLM）。

用已跑完的 refine_cv_qwen_panel（医生自由词）trace，重放每一条 [TEST]：
  - raw：原始自由词能否命中该例规范面板里的检查名；
  - norm：归一化后（按 ' / ' 拆组件）能否命中。
对比命中率，量化「医生自由词 → 查表找规范词」能救回多少措辞不匹配。

命中判定：组件归一化后与某条面板事实 test 名相等，或互为子串（长度≥5）。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exam_normalize import ExamNormalizer, _norm

ROOT = Path("/workspace/medical_llm_exp")
PANEL_DIR = ROOT / "data" / "exam_panels_canonical"
REC_DIR = (
    ROOT / "benchmarks" / "eid-benchmark" / "results" / "medqa"
    / "refine_cv_qwen_panel" / "qwen-plus" / "10_turns" / "record"
)
CATALOG = ROOT / "data" / "exam_catalog.json"


def extract_tests(content: str) -> list[str]:
    out = []
    for m in re.finditer(r"\[TEST\]\s*(.+)", content):
        t = m.group(1).strip().split("\n")[0].strip()
        out.append(t)
    return out


def panel_test_names(case_id: str) -> list[str]:
    fp = PANEL_DIR / f"{case_id}.json"
    if not fp.exists():
        return []
    d = json.load(open(fp))
    return [it.get("test", "") for it in d.get("panel", {}).get("exam_facts", [])]


def _hit_one(s: str, names: list[str]) -> bool:
    sn = _norm(s)
    if len(sn) < 3:
        return False
    for n in names:
        nn = _norm(n)
        if not nn:
            continue
        if sn == nn:
            return True
        if len(sn) >= 5 and len(nn) >= 5 and (sn in nn or nn in sn):
            return True
    return False


def hits(request: str, names: list[str]) -> bool:
    for comp in request.split(" / "):
        if _hit_one(comp.strip(), names):
            return True
    return False


def main():
    nz = ExamNormalizer(CATALOG)
    total = 0
    raw_hit = 0
    norm_hit = 0
    rescued = 0
    lost = 0
    passthrough = 0
    examples_rescue = []
    examples_lost = []

    for fp in sorted(REC_DIR.glob("*.json")):
        d = json.load(open(fp))
        cid = d.get("case_id", "")
        names = panel_test_names(cid)
        if not names:
            continue
        for e in d.get("trace", []):
            if e.get("role_id") != "doctor":
                continue
            for t in extract_tests(e.get("content", "")):
                total += 1
                h_raw = hits(t, names)
                n = nz.normalize(t)
                h_norm = hits(n, names)
                raw_hit += h_raw
                norm_hit += h_norm
                if n.strip() == t.strip():
                    passthrough += 1
                if not h_raw and h_norm:
                    rescued += 1
                    if len(examples_rescue) < 30:
                        examples_rescue.append((cid, t, n))
                elif h_raw and not h_norm:
                    lost += 1
                    if len(examples_lost) < 30:
                        examples_lost.append((cid, t, n))

    print("=" * 96)
    print("离线评估：医生自由词 → 规范名归一化（refine_cv_qwen_panel 重放）")
    print("=" * 96)
    print(f"  [TEST] 请求总数        : {total}")
    print(f"  透传(未归一化)         : {passthrough}  ({passthrough/total*100:.1f}%)")
    print(f"  raw 命中规范面板       : {raw_hit}  ({raw_hit/total*100:.1f}%)")
    print(f"  norm 命中规范面板      : {norm_hit}  ({norm_hit/total*100:.1f}%)")
    print(f"  ✅ 归一化救回(未→中)   : {rescued}")
    print(f"  ❌ 归一化丢失(中→未)   : {lost}")
    print()
    if examples_rescue:
        print("救回示例（raw 未命中 → norm 命中）:")
        for cid, t, n in examples_rescue:
            print(f"  [{cid}] raw: {t[:90]}")
            print(f"        norm: {n[:90]}")
    print()
    if examples_lost:
        print("丢失示例（raw 命中 → norm 未命中）:")
        for cid, t, n in examples_lost:
            print(f"  [{cid}] raw: {t[:90]}")
            print(f"        norm: {n[:90]}")
    print()


if __name__ == "__main__":
    main()

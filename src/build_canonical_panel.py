"""把面板检查名对齐到全局规范目录（P1 措辞对齐的「面板侧」）。

面板里的 test 字段命名混乱（1387 unique 名），对齐到 data/exam_catalog.json 的规范名：
  - test 命中映射表 → 替换为规范名；
  - test 是「结果描述型」（result 为空、描述塞在 test 里）→ 把描述搬到 result，test 用规范名；
  - 未映射的 → 保留原样（result 为空则把 test 搬到 result，至少不丢信息）。

产出：
  - data/exam_panels_canonical/*.json（对齐后的面板）
  - benchmarks/eid-benchmark/data/agentclinic_medqa_panel_canonical.jsonl（新数据集）
"""
import json
import re
import glob
from pathlib import Path

PANEL_DIR = Path("data/exam_panels")
CANON_DIR = Path("data/exam_panels_canonical")
SRC_DATASET = Path("benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl")
DST_DATASET = Path("benchmarks/eid-benchmark/data/agentclinic_medqa_panel_canonical.jsonl")

MAP = json.load(open("data/exam_catalog_map.json"))
CATALOG_ITEMS = json.load(open("data/exam_catalog.json"))["items"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


# 规范名归一化索引
NAME_NORM = {norm(it["name"]): it["name"] for it in CATALOG_ITEMS}
# 别名归一化索引（多个冲突时保留 name 更短/更具体的）
ALIAS_NORM = {}
for it in CATALOG_ITEMS:
    for a in it.get("aliases", []) + [it["name"]]:
        k = norm(a)
        if k not in ALIAS_NORM or len(it["name"]) < len(ALIAS_NORM[k]):
            ALIAS_NORM[k] = it["name"]
# 规范名 token 索引（用于 token 重叠匹配，优先 name 本身与脏名重叠更多、name 更短）
NAME_ITEMS = [(tokens(it["name"]), it["name"]) for it in CATALOG_ITEMS]


def canonicalize(test: str):
    """脏检查名 -> 规范名。

    优先级：
      1. 规范名本身归一化精确匹配；
      2. token 重叠匹配（选 name 与脏名共享 token 最多者，tie 时 name 更短）——
         这一步能把 'Blood_Urea_Nitrogen' 正确归到 'Blood Urea Nitrogen (BUN)'
         而非 'Comprehensive Metabolic Panel (CMP)'（CMP 的别名混入了组分脏写法）；
      3. 别名归一化精确匹配。
    """
    dn = norm(test)
    if dn in NAME_NORM:
        return NAME_NORM[dn]
    dt = tokens(test)
    best = None
    best_score = -1
    for nt, name in NAME_ITEMS:
        ov = len(dt & nt)
        if ov > best_score or (ov == best_score and best and len(name) < len(best)):
            best_score = ov
            best = name
    if best is not None and best_score >= 1:
        return best
    if dn in ALIAS_NORM:
        return ALIAS_NORM[dn]
    return None


def align_item(it: dict, stats: dict) -> dict:
    t = it.get("test", "")
    r = it.get("result", "")
    canon = canonicalize(t)
    if canon is None:
        stats["unmapped"] += 1
        if not r.strip():
            it["result"] = t  # 描述搬到 result，避免信息丢失
            stats["desc_moved"] += 1
        return it
    stats["mapped"] += 1
    if not r.strip():
        it["result"] = t  # 描述搬到 result
        stats["desc_moved"] += 1
    it["test"] = canon
    return it


def merge_items(items):
    """同名 test 合并 result（多个结果用 ' | ' 连接，去重），保持首次出现顺序。"""
    merged = {}
    order = []
    for it in items:
        t = it.get("test", "")
        r = it.get("result", "")
        if t not in merged:
            merged[t] = dict(it)
            order.append(t)
        else:
            if r and r not in merged[t]["result"]:
                merged[t]["result"] = (merged[t]["result"] + " | " + r) if merged[t]["result"] else r
    return [merged[t] for t in order]


def panel_to_exam_facts(panel_items):
    lines = []
    for i, item in enumerate(panel_items, 1):
        t = item.get("test", "")
        r = item.get("result", "")
        line = f"{t}: {r}" if r else t
        lines.append(f"{i}. {line}")
    return lines


def main():
    stats = {"mapped": 0, "unmapped": 0, "desc_moved": 0}
    CANON_DIR.mkdir(parents=True, exist_ok=True)

    panels = {}
    for fp in sorted(glob.glob(str(PANEL_DIR / "agentclinic_medqa_*.json"))):
        d = json.load(open(fp))
        for it in d["panel"]["exam_facts"]:
            align_item(it, stats)
        d["panel"]["exam_facts"] = merge_items(d["panel"]["exam_facts"])
        panels[d["case_id"]] = d["panel"]["exam_facts"]
        (CANON_DIR / f"{d['case_id']}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2))

    print(f"面板项对齐：mapped={stats['mapped']}  unmapped={stats['unmapped']}  "
          f"desc_moved={stats['desc_moved']}")

    # 生成 canonical 数据集
    n_replaced = 0
    with SRC_DATASET.open() as fh, DST_DATASET.open("w") as out:
        for line in fh:
            d = json.loads(line)
            cid = d.get("id")
            if cid in panels:
                d["atomic_facts"]["exam_facts"] = panel_to_exam_facts(panels[cid])
                n_replaced += 1
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"已替换 {n_replaced} 例 exam_facts -> {DST_DATASET}")

    # 未映射的 test 名（供后续人工看是否要补）
    unmapped_names = set()
    for fp in sorted(glob.glob(str(CANON_DIR / "agentclinic_medqa_*.json"))):
        d = json.load(open(fp))
        for it in d["panel"]["exam_facts"]:
            # 已对齐的 test 名都是规范名；未对齐的保留了原 test（无从区分）
            pass
    print(f"canonical 面板 -> {CANON_DIR}")


if __name__ == "__main__":
    main()

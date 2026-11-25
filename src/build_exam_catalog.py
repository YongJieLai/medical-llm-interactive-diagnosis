"""构建「全局规范检查目录」（exam catalog），用于双端对齐措辞。

背景（P1 措辞对齐）：
  面板里的检查名是 LLM 自由发挥生成的，命名极其混乱（实测 2355 个检查项有 1387 个 unique 名）：
    - 同一检查多种写法：ALT / Alanine aminotransferase (ALT) / Alanine_aminotransferase / 'ALT is 34 U/L.'
    - 下划线 vs 空格、缩写 vs 全称混用
    - 「结果描述」混进检查名：'A 2.5-cm firm nodule...'、'Abdomen is soft'、'Blood pressure 130/85'
  导致 reporter 按 verbatim 检索时，医生问的词和面板里的词对不上 → 假失败（Not specified 492 次）。

解法（全局目录双端对齐，用户拍板）：
  建一份「跨病例全局固定」的规范检查目录，让「医生问的词 == 面板里的词」：
    1. 医生侧：prompt 附目录，医生只从目录里选规范检查名（消除「医生问了面板没有的词」）；
    2. 面板侧：面板的 test 字段对齐到目录规范名（消除「同一检查不同词」）；
    3. reporter 保持 verbatim，措辞已对齐，verbatim 规则就能命中。

为什么公平、不泄露：
  目录跨整个数据集恒定（同一份给所有 case），不含「这一题该查什么」的信息；
  医生拿到的是「这家虚拟医院能查哪些检查」的公开清单，而非「本题的事实表」。

实现：
  从 60 例面板聚合 unique 检查名 → 分批喂 LLM 规范化（合并重复、结果描述归位、给别名）
  → 合并去重 → 输出 data/exam_catalog.json + data/exam_catalog_map.json（脏名→规范名）。
"""

import json
import re
import sys
import time
import argparse
import glob
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_llm, chat  # noqa: E402

PANEL_DIR = Path("data/exam_panels")
CATALOG_OUT = Path("data/exam_catalog.json")
MAP_OUT = Path("data/exam_catalog_map.json")

NORMALIZE_PROMPT = """你是一名医学检验/检查目录标准化专家。下面是一批从「交互式诊断 benchmark」的事实库里抽取出来的「检查名/检验名/体格检查描述」，它们的命名非常混乱（下划线/空格/大小写不统一、缩写与全称混用、同一检查多种写法、甚至有「结果描述」混进了检查名字段）。

请你把它们规范化成一份「规范检查目录」的条目，规则如下：

1. 【合并同义】同一项检查的多种写法必须合并成一条规范名（例如 ALT / Alanine aminotransferase (ALT) / Alanine_aminotransferase / ALT is 34 U/L. → 都归为 "Alanine Aminotransferase (ALT)"）。
2. 【结果描述归位】如果某条文本不是「检查名」而是「检查的发现/结果描述」（例如 'A 2.5-cm firm nodule on the neck'、'Abdomen is soft'、'Blood pressure 130/85 mmHg'、'grade 2/6 holosystolic murmur'），不要把它当作检查名，而是把它归到对应的体格检查/影像检查项下（例如归为 "Physical examination - skin inspection"、"Abdominal palpation"、"Blood pressure measurement"、"Cardiac auscultation"），并在 aliases 里保留该原文本。
3. 【规范命名】规范名用英文 Title Case（每个实词首字母大写），去掉下划线改为空格，保留最有识别度的常见缩写（如 ALT、CBC、MRI、CT、PCR），缩写放括号里。
4. 【分类】每项给 category，只能是以下之一：vital_signs / physical_exam / laboratory / imaging / pathology / other。
5. 【别名】aliases 列出该批输入里所有归到这一项的原写法，以及常见同义缩写（不要编造不存在的别名）。
6. 【禁止兜底】不要创建 "Other"/"Clinical Assessment" 之类的兜底条目来收纳大量无法归类的输入；每个输入都必须归到具体的检查条目。同一批输入里的每一条原写法，只能出现在一个条目的 aliases 中（不要既在具体条目里列、又塞进某个兜底条目）。

【输入的一批检查名/描述】
{names}

【输出格式】只输出 JSON，不要任何其他文字：
{{"items": [{{"name": "规范检查名", "category": "分类", "aliases": ["原写法1", "原写法2"]}}]}}
"""


def load_unique_names(panel_dir: Path) -> list:
    names = []
    for fp in sorted(glob.glob(str(panel_dir / "agentclinic_medqa_*.json"))):
        d = json.load(open(fp))
        for it in d.get("panel", {}).get("exam_facts", []):
            names.append(it.get("test", ""))
    # 去重（保留首次出现顺序）
    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def normalize_batch(client, model, names_batch: list) -> list:
    """LLM 规范化一批检查名，返回 items 列表。"""
    msg = NORMALIZE_PROMPT.format(names=json.dumps(names_batch, ensure_ascii=False, indent=0))
    raw = chat(client, model, [{"role": "user", "content": msg}],
               temperature=0.0, max_tokens=8192, json_mode=True)
    if isinstance(raw, str):
        print(f"    [batch] JSON 解析失败，原始返回前 300 字:\n{raw[:300]}")
        return []
    return raw.get("items", [])


def merge_items(batches_items: list) -> list:
    """跨批合并：同名项合并 aliases，规范化 name 去重。"""
    def canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    merged = {}
    for items in batches_items:
        for it in items:
            name = it.get("name", "").strip()
            if not name:
                continue
            key = canon(name)
            if key not in merged:
                merged[key] = {"name": name, "category": it.get("category", "other"),
                               "aliases": set()}
            merged[key]["aliases"].update(a.strip() for a in it.get("aliases", []) if a and a.strip())
            # 类别冲突时取更具体的（physical_exam/vital_signs 优先于 other）
            cat = it.get("category", "other")
            if merged[key]["category"] == "other" and cat != "other":
                merged[key]["category"] = cat
    out = []
    for key, v in merged.items():
        out.append({"name": v["name"], "category": v["category"],
                    "aliases": sorted(v["aliases"])})
    # 按 category + name 排序，便于阅读
    order = {"vital_signs": 0, "physical_exam": 1, "laboratory": 2, "imaging": 3, "pathology": 4, "other": 5}
    out.sort(key=lambda x: (order.get(x["category"], 9), x["name"].lower()))
    out = dedup_aliases(out)
    return out


def dedup_aliases(items: list) -> list:
    """冲突 alias 消解：每个 alias 只保留在「最具体」（非 other）的条目里；
    纯兜底条目（name 含 Other 且 aliases 已被占用）直接丢弃。"""
    order = {"vital_signs": 0, "physical_exam": 1, "laboratory": 2, "imaging": 3, "pathology": 4, "other": 5}
    items = sorted(items, key=lambda x: order.get(x["category"], 9))
    claimed = {}  # alias_lower -> item name
    out = []
    for it in items:
        keep = []
        for a in it.get("aliases", []):
            k = a.lower()
            if k not in claimed:
                claimed[k] = it["name"]
                keep.append(a)
        it["aliases"] = sorted(keep)
        if it["category"] == "other" and "other" in it["name"].lower() and not keep:
            continue  # 纯兜底条目，内容都已被具体条目 claim，丢弃
        out.append(it)
    # 恢复原来的 category+name 排序
    out.sort(key=lambda x: (order.get(x["category"], 9), x["name"].lower()))
    return out


def build_map(catalog_items: list, unique_names: list) -> dict:
    """从目录的 aliases 反推「脏名 -> 规范名」映射；未覆盖的脏名尝试精确/包含匹配。"""
    m = {}
    # 先建 alias 索引
    alias2name = {}
    for it in catalog_items:
        n = it["name"]
        alias2name[n.lower()] = n
        for a in it.get("aliases", []):
            alias2name[a.lower()] = n
    def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
    # 规范名归一化索引，用于兜底模糊匹配
    name_norm = {norm(it["name"]): it["name"] for it in catalog_items}
    uncovered = []
    for dirty in unique_names:
        low = dirty.lower()
        if low in alias2name:
            m[dirty] = alias2name[low]
            continue
        # 兜底：归一化后精确匹配规范名
        dn = norm(dirty)
        if dn in name_norm:
            m[dirty] = name_norm[dn]
            continue
        uncovered.append(dirty)
    return m, uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="qwen")
    ap.add_argument("--batch", type=int, default=200, help="每批喂给 LLM 的检查名数量")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 unique 名（smoke 用）")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    client, model = get_llm(args.gen)
    names = load_unique_names(PANEL_DIR)
    if args.limit:
        names = names[:args.limit]
    print(f"unique 检查名: {len(names)}")

    # 分批
    batches = [names[i:i + args.batch] for i in range(0, len(names), args.batch)]
    print(f"分 {len(batches)} 批（每批 ~{args.batch} 名）规范化...")

    all_items = []
    t0 = time.time()
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(normalize_batch, client, model, b): i for i, b in enumerate(batches)}
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                items = fut.result()
                print(f"  [batch {i+1}/{len(batches)}] 输出 {len(items)} 个规范条目")
                all_items.append(items)
    else:
        for i, b in enumerate(batches):
            items = normalize_batch(client, model, b)
            print(f"  [batch {i+1}/{len(batches)}] 输出 {len(items)} 个规范条目")
            all_items.append(items)
    print(f"规范化耗时 {time.time()-t0:.0f}s")

    catalog_items = merge_items(all_items)
    print(f"合并去重后规范目录条目: {len(catalog_items)}")

    # 类别分布
    from collections import Counter
    cat = Counter(it["category"] for it in catalog_items)
    print("类别分布:", dict(cat))

    CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_OUT.write_text(json.dumps({"items": catalog_items}, ensure_ascii=False, indent=2))
    print(f"已保存目录 -> {CATALOG_OUT}")

    m, uncovered = build_map(catalog_items, names)
    MAP_OUT.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"已保存映射 -> {MAP_OUT}（映射 {len(m)}/{len(names)}，未覆盖 {len(uncovered)}）")
    if uncovered:
        print("未覆盖样例（前 20）:")
        for u in uncovered[:20]:
            print("   ", repr(u[:70]))


if __name__ == "__main__":
    main()

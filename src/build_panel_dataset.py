"""把离线生成的完整检查面板转成 exam_facts 格式，生成新数据集。

面板（data/exam_panels/*.json 里的 panel.exam_facts）→ exam_facts 列表（"N. test: result"），
替换原数据集的 atomic_facts.exam_facts，输出新 jsonl。
"""
import json
from pathlib import Path

PANEL_DIR = Path("data/exam_panels")
SRC = Path("benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl")
DST = Path("benchmarks/eid-benchmark/data/agentclinic_medqa_panel.jsonl")


def panel_to_exam_facts(panel_items):
    lines = []
    for i, item in enumerate(panel_items, 1):
        t = item.get("test", "")
        r = item.get("result", "")
        line = f"{t}: {r}" if r else t
        lines.append(f"{i}. {line}")
    return lines


def main():
    panels = {}
    for fp in sorted(PANEL_DIR.glob("agentclinic_medqa_*.json")):
        d = json.loads(fp.read_text())
        panels[d["case_id"]] = d["panel"]["exam_facts"]
    print(f"已加载 {len(panels)} 例面板")

    n_replaced = 0
    with SRC.open() as fh, DST.open("w") as out:
        for line in fh:
            d = json.loads(line)
            cid = d.get("id")
            if cid in panels:
                ef = panel_to_exam_facts(panels[cid])
                d["atomic_facts"]["exam_facts"] = ef
                n_replaced += 1
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"已替换 {n_replaced} 例的 exam_facts，输出 -> {DST}")


if __name__ == "__main__":
    main()

"""检查 HPOA（disease-phenotype）对 MedQA 58 答案病的覆盖。

回答：HPO 表型 grounding 能覆盖多少 MedQA 病？这是决定 P2-PrimeKG 真实版
是否值得直接跑 30 例的关键前置检查（HPOA 偏罕见病，常见病可能覆盖差）。

运行：python3 src/check_hpoa_coverage.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from primekg_ground import PhenotypeGrounder  # noqa: E402


def main():
    hpoa = REPO / "data" / "hpo" / "phenotype.hpoa"
    hp = REPO / "data" / "hpo" / "hp.json"
    g = PhenotypeGrounder(str(hpoa), str(hp) if hp.exists() else None)

    kb = json.loads((REPO / "data" / "ddx_knowledge.json").read_text())
    answers = list(kb.keys())

    matched, unmatched = [], []
    for a in answers:
        key = g._match_disease(a)
        if key:
            matched.append((a, key, len(g._disease_to_phenos[key])))
        else:
            unmatched.append(a)

    print(f"HPOA 疾病条目数: {len(g._disease_to_phenos)}")
    print(f"MedQA 58 答案病覆盖: {len(matched)}/{len(answers)} "
          f"({100*len(matched)/len(answers):.1f}%)\n")

    print("--- 匹配（病名 -> HPOA 病名，表型数）---")
    for a, key, nphe in matched:
        print(f"  {a:55s} -> {key:55s} ({nphe} 表型)")

    print("\n--- 未匹配 ---")
    for a in unmatched:
        print(f"  ✗ {a}")


if __name__ == "__main__":
    main()

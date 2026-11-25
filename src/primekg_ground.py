"""PrimeKG / HPO 不泄题 grounding：disease -> phenotype 表型清单。

替代 P2 oracle KB（`data/ddx_knowledge.json`，病集合==答案病，泄题）。
数据源是 HPO 官方的 disease-phenotype annotation（phenotype.hpoa），即
PrimeKG 里 disease-phenotype 边的原始来源，独立于 MedQA 答案标签。

用法：
    from primekg_ground import PhenotypeGrounder
    g = PhenotypeGrounder("data/hpo/phenotype.hpoa", "data/hpo/hp.json")
    snippet = g.retrieve("myasthenia gravis")   # -> 表型清单文本，或 ""

口径与 oracle 的差异（重要）：
    oracle KB 注入的是 differential_diagnoses + distinguishing_feature（相似病
    + 怎么区分），PrimeKG/HPO 只有 disease->phenotype（该病有什么表型）。所以
    本模块是「表型清单 grounding」，不提供鉴别区分点——那属于运行时 LLM 的职责。
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class PhenotypeGrounder:
    """Load HPOA + HPO names, expose disease -> phenotype-list retrieval."""

    def __init__(self, hpoa_path: str, hp_json_path: str | None = None) -> None:
        self._disease_to_phenos: dict[str, list[dict]] = self._load_hpoa(hpoa_path)
        self._hpo_names: dict[str, str] = self._load_hp_names(hp_json_path)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_hpoa(path: str) -> dict[str, list[dict]]:
        """Parse phenotype.hpoa -> {normalised_disease_name: [{hpo_id, qualifier,
        frequency}]}. Keeps the longest disease name per normalised key on
        collision so verbose names aren't lost."""
        out: dict[str, list[dict]] = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line or line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 4:
                        continue
                    _, disease_name, qualifier, hpo_id = cols[0], cols[1], cols[2], cols[3]
                    frequency = cols[7] if len(cols) > 7 else ""
                    norm = PhenotypeGrounder._norm_med(disease_name)
                    if not norm or not hpo_id.startswith("HP:"):
                        continue
                    out.setdefault(norm, []).append(
                        {"hpo_id": hpo_id, "qualifier": qualifier, "frequency": frequency}
                    )
        except FileNotFoundError:
            return {}
        return out

    @staticmethod
    def _load_hp_names(path: str | None) -> dict[str, str]:
        """Parse hp.json -> {HP:xxxxxxx: name}."""
        if not path:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        out: dict[str, str] = {}
        graphs = data.get("graphs", []) if isinstance(data, dict) else []
        for g in graphs:
            for n in g.get("nodes", []):
                i = n.get("id", "")
                lbl = n.get("lbl") or n.get("meta", {}).get("synonyms", [None])[0]
                if i.startswith("HP:") and lbl:
                    out[i] = str(lbl)
        return out

    @staticmethod
    def _norm_med(s: str) -> str:
        s = re.sub(r"['\u2019]s\b", " ", s or "")
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"[^a-z0-9]+", " ", s.lower())
        return " ".join(s.split())

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _match_disease(self, diagnosis: str) -> str | None:
        """Fuzzy-match `diagnosis` to the longest HPOA disease name."""
        d = self._norm_med(diagnosis)
        if not d:
            return None
        best_key, best_len = None, 0
        for key in self._disease_to_phenos:
            if key in d or d in key:
                if len(key) > best_len:
                    best_key, best_len = key, len(key)
        return best_key

    def retrieve(self, diagnosis: str, top_k: int = 10) -> str:
        """Return a compact phenotype-list snippet for `diagnosis`, or ''."""
        key = self._match_disease(diagnosis)
        if key is None:
            return ""
        phenos = self._disease_to_phenos[key]
        # 优先阳性（qualifier 空/非 NOT）、高频（1/1 等）表型；去重保序
        def freq_rank(f: str) -> int:
            try:
                num, den = f.split("/")
                return int(num) * 100 // int(den)
            except Exception:
                return 0

        present = [p for p in phenos if p["qualifier"].upper() != "NOT"]
        absent = [p for p in phenos if p["qualifier"].upper() == "NOT"]
        present.sort(key=lambda p: -freq_rank(p["frequency"]))
        seen: set[str] = set()
        lines: list[str] = []
        for p in present:
            name = self._hpo_names.get(p["hpo_id"], p["hpo_id"])
            if name in seen:
                continue
            seen.add(name)
            lines.append(f"- {name}")
            if len(lines) >= top_k:
                break
        if not lines:
            return ""
        header = (
            f"Typical phenotypes of {key} (from HPO, authoritative):"
        )
        body = "\n".join(lines)
        if absent:
            neg = ", ".join(
                self._hpo_names.get(p["hpo_id"], p["hpo_id"])
                for p in absent[:5]
            )
            body += f"\n(Notably NOT associated: {neg})"
        return f"{header}\n{body}"


def main() -> None:
    """CLI smoke test."""
    import sys

    hpoa = sys.argv[1] if len(sys.argv) > 1 else "data/hpo/phenotype.hpoa"
    hp = sys.argv[2] if len(sys.argv) > 2 else "data/hpo/hp.json"
    g = PhenotypeGrounder(hpoa, hp)
    print(f"loaded diseases: {len(g._disease_to_phenos)}")
    for dx in ["myasthenia gravis", "jersey finger", "asthma", "frozen shoulder"]:
        print("=" * 60)
        print("QUERY:", dx)
        s = g.retrieve(dx)
        print(s if s else "(no match)")


if __name__ == "__main__":
    main()

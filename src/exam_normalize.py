"""医生自由词 → 规范检查名 的确定性归一化器（P1 措辞对齐 · reporter 端）。

用户方案：
  医生**不看目录**自由开 [TEST] → 开完后用「规范检查目录」把医生的自由词翻译成规范名
  → 把规范名交给 Measurement（reporter）查事实表。

效果拆解：
  - 医生端无目录 → 不泄露检查空间、不被目录带偏（规避"公开目录给医生有害"负结果）；
  - reporter 端措辞对齐 → 消除"问对但词对不上"导致的假阴性 / 假失败（P1）。

匹配原则（**precision 优先**）：宁可透传（诚实 Not specified），绝不误映射（误映射会把
医生带偏）。因此：
  - 子串包含匹配**只用规范名本身**（657 个干净、具体的检查名），绝不用别名做子串——
    别名里混着 "csf"/"pcr"/"reflex" 等泛词与结果描述，会假阳性（JC virus PCR → EBV PCR）；
  - 别名只用于「整串归一化精确匹配」（医生恰好照写别名，罕见且安全）；
  - token 重叠兜底要求「唯一最优 且 ≥2 个实义 token」，歧义时透传。

匹配顺序：
  1. 整串精确匹配（规范名或别名）；
  2. 规范名作为子串出现在请求里（可多个，支持复合请求 "CBC + peripheral smear"）；
  3. 请求作为子串出现在规范名里（缩写 "cbc" ⊂ "completebloodcountcbc"）；
  4. token 重叠兜底（唯一最优、≥2 实义 token，如 "mri brain"→"Brain MRI"）；
  5. 透传原串。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# 通用/无信息 token：token 重叠兜底时忽略。
_STOPWORDS = {
    "test", "tests", "testing", "analysis", "panel", "level", "levels",
    "review", "assessment", "evaluation", "examination", "exam", "study",
    "with", "and", "for", "of", "the", "a", "an", "to", "in", "on", "or",
    "is", "are", "be", "by", "as", "at", "from", "plus", "including",
    "include", "includes", "if", "indicated", "available", "performed",
    "right", "left", "bilateral", "serial", "repeat", "routine", "full",
    "complete", "profile", "assay", "measurement", "monitoring", "view",
    "views", "viewing", "obtained", "using", "following", "given",
}


class ExamNormalizer:
    """把自由文本检查请求翻译为规范检查名（可能多个，用 ' / ' 连接）。"""

    def __init__(self, catalog_path: str | Path) -> None:
        catalog = json.load(open(catalog_path, encoding="utf-8"))["items"]

        self._name_norm: dict[str, str] = {}   # 规范名归一化 -> 规范名
        self._alias_norm: dict[str, str] = {}  # 别名归一化 -> 规范名（仅精确匹配）
        self._name_entries: list[tuple[str, str]] = []  # (归一化名, 规范名)，长优先
        self._name_tokens: list[tuple[set[str], str]] = []  # (实义 token, 规范名)

        for it in catalog:
            name = it["name"]
            nn = _norm(name)
            self._name_norm[nn] = name
            self._name_entries.append((nn, name))
            # token 集合丢弃单字符（如 "b"）与停用词，避免 "serum"+"b" 这类假重叠。
            toks = {
                t for t in re.findall(r"[a-z0-9]+", name.lower())
                if len(t) >= 2 and t not in _STOPWORDS
            }
            self._name_tokens.append((toks, name))
            for a in it.get("aliases", []):
                k = _norm(a)
                if k and k not in self._alias_norm:
                    self._alias_norm[k] = name
        self._name_entries.sort(key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _ordered(names: list[str], request: str) -> str:
        """去重、去冗余（A 是 B 的子串则丢 A 留 B）、按出现位置排序；多个 ' / ' 连接。"""
        seen: list[str] = []
        for n in names:
            if n and n not in seen:
                seen.append(n)
        if not seen:
            return request
        # 去冗余：若某规范名被另一个更长规范名完全包含（归一化后），丢弃短的。
        norm_map = {n: _norm(n) for n in seen}
        kept = []
        for n in seen:
            nn = norm_map[n]
            if any(n != m and nn in norm_map[m] for m in seen):
                continue
            kept.append(n)
        if kept:
            seen = kept
        rn = _norm(request)
        seen.sort(key=lambda c: rn.find(norm_map[c]) if norm_map[c] in rn else 10**9)
        return " / ".join(seen) if len(seen) > 1 else seen[0]

    def normalize(self, test: str) -> str:
        """自由词 -> 规范名；未命中（或歧义）透传原串。"""
        if not test or not test.strip():
            return test
        rn = _norm(test)
        if not rn:
            return test

        # 1) 整串精确匹配（规范名 / 别名）
        if rn in self._name_norm:
            return self._name_norm[rn]
        if rn in self._alias_norm:
            return self._alias_norm[rn]

        # 2) 规范名作为子串出现在请求里（复合请求可命中多个）
        contained = [name for nn, name in self._name_entries if len(nn) >= 5 and nn in rn]
        if contained:
            return self._ordered(contained, test)

        # 3) 请求作为子串出现在规范名里（缩写）
        if len(rn) >= 3:
            rev = [name for nn, name in self._name_entries if rn in nn]
            if rev:
                return self._ordered(rev, test)

        # 4) token 重叠兜底：唯一最优 且 ≥2 实义 token（丢弃单字符 token）
        rtoks = {
            t for t in re.findall(r"[a-z0-9]+", test.lower())
            if len(t) >= 2 and t not in _STOPWORDS
        }
        scored = []
        for ntoks, name in self._name_tokens:
            ov = len(rtoks & ntoks)
            if ov >= 2:
                scored.append((ov, -len(name), name))
        if scored:
            scored.sort(reverse=True)
            top_score = scored[0][0]
            if len(scored) == 1 or scored[1][0] < top_score:
                return scored[0][2]

        # 5) 透传
        return test

    @classmethod
    def from_env(cls) -> "ExamNormalizer | None":
        """按环境变量 EXAM_CATALOG 加载；未设置或文件缺失返回 None（关闭归一化）。"""
        catalog_path = os.getenv("EXAM_CATALOG", "")
        if not catalog_path or not Path(catalog_path).exists():
            return None
        return cls(catalog_path)

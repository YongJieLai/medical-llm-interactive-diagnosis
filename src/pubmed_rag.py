"""PubMed E-utilities 检索模块：为 CEV-GR 的 verifier 提供不泄题的 grounding。

背景（rag-ceiling-analysis.md）：P2 的 oracle KB（data/ddx_knowledge.json）病集合
== 60 题答案病，属测试集先验泄漏，+6.7pp 只能作知识上界。HF 教科书语料在本沙箱
不可达，但 NCBI E-utilities（PubMed/Bookshelf）可达 —— 这正是 MEDDxAgent 所用的
同类通用知识源，独立于答案标签，不泄题。

用法：
    from pubmed_rag import ground
    snippet = ground("mallet finger", hints="distal interphalangeal joint")

设计：
    esearch(db, term) -> PMID 列表
    efetch_abstracts(pmids) -> 精简摘要文本（标题 + 摘要开头）
    ground(diagnosis, hints) -> 检索诊断名（+可选症状/部位提示），返回精简文本

E-utilities 无 key 限 3 req/s，内置 0.4s 节流。
"""
import json
import time
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def esearch(term: str, db: str = "pubmed", retmax: int = 3) -> list[str]:
    """按 term 检索，返回 ID 列表（PMID 或 book UID）。"""
    q = urllib.parse.urlencode(
        {"db": db, "term": term, "retmax": retmax, "retmode": "json"}
    )
    try:
        d = json.loads(_get(f"{EUTILS}/esearch.fcgi?{q}"))
        return d.get("esearchresult", {}).get("idlist", [])
    except Exception:  # noqa: BLE001 - 检索失败返回空
        return []


def efetch_abstracts(pmids: list[str], db: str = "pubmed", max_chars: int = 1800) -> str:
    """抓取 ID 列表的摘要文本，截断到 max_chars。"""
    if not pmids:
        return ""
    q = urllib.parse.urlencode(
        {"db": db, "id": ",".join(pmids), "rettype": "abstract", "retmode": "text"}
    )
    try:
        txt = _get(f"{EUTILS}/efetch.fcgi?{q}")
        return txt[:max_chars]
    except Exception:  # noqa: BLE001
        return ""


def ground(
    diagnosis: str,
    hints: str = "",
    db: str = "pubmed",
    retmax: int = 2,
    max_chars: int = 1800,
) -> str:
    """检索一个诊断名（+可选提示词），返回精简的临床知识片段。

    空结果或网络失败返回 ''，由调用方决定是否降级（fail-open 到 verifier 自身知识）。
    """
    diagnosis = (diagnosis or "").strip()
    if not diagnosis:
        return ""
    term = diagnosis
    if hints:
        term = f"{diagnosis} AND ({hints})"
    pmids = esearch(term, db=db, retmax=retmax)
    if not pmids:
        return ""
    time.sleep(0.4)  # E-utilities 节流
    return efetch_abstracts(pmids, db=db, max_chars=max_chars)

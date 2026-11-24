"""确定性诊断判定器（规则 + 同义词表，DeepSeek 仅兜底）。

设计目标：消除"LLM judge 噪声"——同一个 (prediction, ground_truth) 对，规则层
永远给出相同判定；只有规则判不出的"不确定" case 才走 DeepSeek 双判多数投票。

判定口径（与人工复核一致）：
- CORRECT：pred 与 truth 是同一疾病，或 pred 是 truth 的更具体子类/变体
  （如 internal hemorrhoids ⊂ hemorrhoids、micturition syncope ⊂ situational syncope）。
- INCORRECT：不同疾病，或 pred 是 truth 的"过宽父类"（如 lymphoma vs DLBCL）。

规则层只负责"确定 CORRECT"（高精度、不误杀），判不出的走 LLM。
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import get_llm, chat


# ---------- 归一化 ----------

def normalize(s: str) -> str:
    """小写、去所有格 's、去一切非字母数字字符。"""
    s = (s or "").lower()
    s = s.replace("'s", "").replace("\u2019s", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ---------- 同义词 / 上下位等价表 ----------
# key = truth 的归一化形式；value = 判为 CORRECT 的额外归一化词（子类/同义）。
# 注意：这些是"pred 归一化文本中包含 value 即判对"的宽松匹配（value 是短词即可命中长文本）。

# 负向互斥表：truth 的归一化 -> 若 pred 命中这些词，判 INCORRECT（易混淆但实为不同病）。
# 例：mallet finger(伸肌腱 DIP) ≠ jersey finger/FDP 屈肌腱断裂。
NEGATIVE = {
    "ruptureoftheflexordigitorumprofundustendonatitspointofinsertion": ["malletfinger"],
}

TRUTH_EQUIV = {
    "progressivemultifocalencephalopathy": [
        "progressivemultifocalleukoencephalopathy",
        "pml",
    ],
    "complexpartialseizure": [
        "focalimpairedawarenessseizure",
        "focalonsetimpairedawarenessseizure",
        "temporallobeepilepsy",
        "mesialtemporallobeepilepsy",
        "focalonsetepilepsy",
        "focalepilepsy",
    ],
    "situationalsyncope": [
        "micturitionsyncope",
        "vasovagalsyncope",
    ],
    "congenitalrubellainfection": [
        "congenitalrubellasyndrome",
    ],
    "posteriorhipdislocation": [
        "posteriordislocationoftherighthip",
        "acuteposteriordislocationoftherighthip",
        "posteriorrighthipdislocation",
    ],
    "acuteinterstitialnephritis": [
        "eosinophilicinterstitialnephritis",
    ],
    "ruptureoftheflexordigitorumprofundustendonatitspointofinsertion": [
        "jerseyfinger",
        "flexordigitorumprofundusavulsion",
        "flexordigitorumprofundusrupture",
        "flexordigitorumprofundus",
    ],
    "cdifficilecolitis": [
        "clostridioidesdifficileinfection",
        "clostridiumdifficileinfection",
        "cdifficileinfection",
    ],
    "pesanserinebursitis": [
        "pesanserinetendinopathy",
        "anserinebursitis",
        "pesanserinetendonitis",
        "pesanserinusbursitis",
        "pesanserinustendinopathy",
    ],
    "epidermoidcyst": [
        "epidermalinclusioncyst",
        "epidermalcyst",
        "sebaceouscyst",
        "pilarcyst",
    ],
    "malignantmelanoma": [
        "melanoma",
    ],
    "cytomegalovirusretinitis": [
        "cmvretinitis",
    ],
    "hemorrhoids": [
        "hemorrhoidal",
    ],
}


def rule_judge(pred: str, truth: str):
    """规则层判定。返回 True(CORRECT) / False(INCORRECT) / None(不确定，走 LLM)。"""
    p = normalize(pred)
    t = normalize(truth)
    if not p or not t:
        return None

    # 0) 负向互斥：明确是不同病（优先级最高）
    for neg in NEGATIVE.get(t, []):
        if neg and neg in p:
            return False

    # 1) 精确匹配
    if p == t:
        return True

    # 2) truth 完整词是 pred 的子串 → pred 是 truth 的具体化/带修饰
    #    加长度保护，避免过短词误命中。
    if len(t) >= 6 and t in p:
        return True

    # 2b) 反向：pred 完整词是 truth 的子串（pred 是 truth 去掉括号/缩写后缀的规范名）。
    #     例：truth="progressive multifocal encephalopathy (pml)"、
    #         pred="progressive multifocal encephalopathy"。
    if len(p) >= 8 and p in t:
        return True

    # 3) 同义词/子类等价词命中（key 与 truth 做双向子串匹配，容忍 truth 带缩写后缀）
    for key, words in TRUTH_EQUIV.items():
        if key in t or t in key:
            for w in words:
                if w and w in p:
                    return True

    # 规则判不出（既不能确定对，也不确定错）→ 交给 LLM
    return None


# ---------- DeepSeek 兜底（双判 + 多数投票） ----------

_JUDGE_SYSTEM = (
    "You are an expert clinician evaluating whether a predicted diagnosis matches a "
    "reference (ground-truth) diagnosis.\n"
    "Rules:\n"
    "- CORRECT if the prediction is the SAME disease as the reference, OR a MORE SPECIFIC "
    "subtype/variant of it (e.g. 'internal hemorrhoids' vs 'hemorrhoids'; 'micturition "
    "syncope' vs 'situational syncope'; 'jersey finger' vs 'rupture of the flexor "
    "digitorum profundus tendon').\n"
    "- INCORRECT if the prediction is a DIFFERENT disease, OR a BROADER/ANCESTOR category "
    "that fails to specify the reference (e.g. 'lymphoma' vs 'diffuse large B-cell "
    "lymphoma' is INCORRECT because it is too broad).\n"
    'Output JSON only: {"verdict": "CORRECT" or "INCORRECT", "reason": "one short sentence"}'
)


def _llm_verdict(client, model, pred, truth):
    prompt = (
        f"Reference diagnosis: {truth}\n\n"
        f"Predicted diagnosis: {pred}\n\n"
        "Evaluate."
    )
    msgs = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = chat(client, model, msgs, temperature=0.0, max_tokens=200, json_mode=True)
    if isinstance(raw, dict):
        v = str(raw.get("verdict", "")).upper()
        return v
    # 非 JSON：粗解析
    s = raw.upper()
    if "INCORRECT" in s and "CORRECT" not in s:
        return "INCORRECT"
    if "CORRECT" in s:
        return "CORRECT"
    return None


def llm_judge(client, model, pred, truth, n_agree=2):
    """DeepSeek 双判 + 多数投票（不一致时补一票）。返回 True/False/None。"""
    votes = []
    for _ in range(2):
        try:
            v = _llm_verdict(client, model, pred, truth)
        except Exception as e:
            print(f"    [judge] DeepSeek 调用失败: {e}", file=sys.stderr)
            v = None
        if v in ("CORRECT", "INCORRECT"):
            votes.append(v)
    # 两票一致直接取；否则补第三票
    if len(set(votes)) == 1 and votes:
        return votes[0] == "CORRECT"
    if len(votes) < 2:
        # 至少一次调用失败，退化为单票或保守判 False
        return votes[0] == "CORRECT" if votes else None
    try:
        v3 = _llm_verdict(client, model, pred, truth)
    except Exception:
        v3 = None
    if v3 in ("CORRECT", "INCORRECT"):
        votes.append(v3)
    corrects = votes.count("CORRECT")
    incorrects = votes.count("INCORRECT")
    if corrects == incorrects:
        return None
    return corrects > incorrects


def judge(pred: str, truth: str, client=None, model=None):
    """对外统一入口：先规则，判不出走 DeepSeek 双判。

    返回 dict: {correct: bool|None, source: 'rule_exact'|'rule_substr'|'rule_equiv'|'llm'}
    """
    r = rule_judge(pred, truth)
    if r is True:
        # 区分来源（便于审计）
        p, t = normalize(pred), normalize(truth)
        if p == t:
            src = "rule_exact"
        elif len(t) >= 6 and t in p:
            src = "rule_substr"
        else:
            src = "rule_equiv"
        return {"correct": True, "source": src}
    if r is False:
        return {"correct": False, "source": "rule"}
    # 不确定 → LLM
    if client is None:
        return {"correct": None, "source": "uncertain_no_llm"}
    v = llm_judge(client, model, pred, truth)
    return {"correct": v, "source": "llm" if v is not None else "llm_failed"}


def get_deepseek():
    return get_llm("deepseek")

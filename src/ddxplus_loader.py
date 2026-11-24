"""DDXPlus 数据加载与病例预处理。

职责：
1. 加载 release_evidences.json / release_conditions.json / test.csv
2. 把结构化证据转成"可问问题"与"患者回答"
3. 分层抽取小规模病例子集，转成实验用 JSON

证据三类：
- B (binary):        在 EVIDENCES 里出现 = 阳性；否则阴性
- C (categorical):   值形如 "E_56_@_6"（数值，如疼痛强度）
- M (multi-choice):  值形如 "E_54_@_V_112"（枚举，value_meaning 里有英文语义）
"""
import json
import ast
import random
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_evidences():
    return json.load(open(DATA_DIR / "release_evidences.json"))


def load_conditions():
    return json.load(open(DATA_DIR / "release_conditions.json"))


def parse_evidence_token(token: str):
    """'E_53' -> ('E_53', None)；'E_54_@_V_112' -> ('E_54', 'V_112')"""
    if "_@_" in token:
        name, val = token.split("_@_", 1)
        return name, val
    return token, None


def evidence_question(ev, name: str) -> str:
    """返回该证据的标准问句（question_en）。"""
    return ev[name]["question_en"]


def evidence_answer(ev, name: str, value):
    """根据证据与值，生成患者对该证据的'阳性'回答文本。

    value: None 表示 binary 阳性；'V_112' 表示 multi-choice；'6' 表示 categorical。
    """
    e = ev[name]
    dt = e["data_type"]
    if dt == "B":
        return "Yes."
    if dt == "C":
        return f"{value}."  # 数值，如疼痛强度 6
    if dt == "M":
        vm = e.get("value_meaning", {})
        if value in vm:
            return f"{vm[value].get('en', value)}."
        return f"{value}."
    return "Yes."


def negative_answer(ev, name: str) -> str:
    """该证据阴性时的回答。"""
    e = ev[name]
    dt = e["data_type"]
    if dt == "B":
        return "No."
    if dt == "C":
        return "Not applicable."
    if dt == "M":
        return "None of these."
    return "No."


def condition_evidence_sets(cond):
    """疾病 -> 该疾病关联的证据名集合（symptoms + antecedents）。"""
    s = set(cond.get("symptoms", {}).keys())
    s |= set(cond.get("antecedents", {}).keys())
    return s


def load_test_cases(ev, cond):
    """加载 test.csv 全部病例为 DataFrame。"""
    df = pd.read_csv(DATA_DIR / "test.csv")
    return df


def sample_cases(df, n_per_class=None, seed=42, max_cases=20):
    """分层抽样病例。

    策略：尽量覆盖不同疾病；每个疾病最多抽 n_per_class 个（默认 1），
    优先选初始证据较少（更需追问）的病例。总数不超过 max_cases。
    """
    rng = random.Random(seed)
    picked = []
    # 随机打乱疾病顺序，避免字母序偏差；组内按 EVIDENCES 数量从少到多排序（证据少的更需追问）
    pathologies = list(df["PATHOLOGY"].unique())
    rng.shuffle(pathologies)
    for patho in pathologies:
        grp = df[df["PATHOLOGY"] == patho].copy()
        grp["_n_ev"] = grp["EVIDENCES"].apply(lambda x: len(ast.literal_eval(x)))
        grp = grp.sort_values("_n_ev")
        k = n_per_class if n_per_class else 1
        for _, row in grp.head(k).iterrows():
            picked.append(row)
            if len(picked) >= max_cases:
                break
        if len(picked) >= max_cases:
            break
    return picked[:max_cases]


def row_to_case(row, ev, cond):
    """把一个 DataFrame 行转成实验用 case dict。"""
    evid_tokens = ast.literal_eval(row["EVIDENCES"])
    evid_values = {}  # 结构化真相：{E_xx: True 或 'V_112' 或 '6'}
    for tok in evid_tokens:
        name, val = parse_evidence_token(tok)
        evid_values[name] = (True if val is None else val)

    initial_tokens = [row["INITIAL_EVIDENCE"]] if isinstance(row["INITIAL_EVIDENCE"], str) else list(row["INITIAL_EVIDENCE"])

    # 初始主诉的自然语言（把初始证据转成陈述，患者开场主动说）
    init_facts = []
    for tok in initial_tokens:
        name, val = parse_evidence_token(tok)
        init_facts.append(evidence_to_statement(ev, name, val))

    return {
        "case_id": f"{row['PATHOLOGY']}_{row.name}",
        "age": int(row["AGE"]),
        "sex": row["SEX"],
        "pathology": row["PATHOLOGY"],
        "differential_diagnosis": ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]),
        "initial_evidence": initial_tokens,
        "initial_facts": init_facts,
        "evidence_values": evid_values,  # 隐藏真相
    }


def evidence_to_statement(ev, name, value):
    """把一个证据转成一句患者自述（用于初始主诉）。"""
    e = ev[name]
    q = e["question_en"].rstrip(" ?")
    dt = e["data_type"]
    if dt == "B":
        # "Do you have a cough?" -> "I have a cough."
        q = q.replace("Do you have ", "I have ").replace("do you have ", "I have ")
        if not q.startswith("I "):
            q = "I have " + q.lower()
        return q + "."
    if dt == "C":
        return f"{q}: {value}."
    if dt == "M":
        vm = e.get("value_meaning", {}).get(value, {}).get("en", value)
        return f"{q}: {vm}."
    return f"{q}."


def build_case_set(max_cases=20, seed=42, out_path=None):
    """主入口：抽样子集并保存为 JSON。"""
    ev = load_evidences()
    cond = load_conditions()
    df = load_test_cases(ev, cond)
    rows = sample_cases(df, n_per_class=1, seed=seed, max_cases=max_cases)
    cases = [row_to_case(r, ev, cond) for r in rows]
    out = {
        "cases": cases,
        "evidences": ev,
        "conditions": cond,
    }
    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "results" / "cases.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(cases)} 个病例到 {out_path}")
    # 打印分布
    from collections import Counter
    print("疾病分布:", dict(Counter(c["pathology"] for c in cases)))
    return out


if __name__ == "__main__":
    out = build_case_set(max_cases=20, seed=42)
    c0 = out["cases"][0]
    print("\n=== 第一个病例预览 ===")
    for k, v in c0.items():
        if k == "evidence_values":
            print(f"  {k}: ({len(v)} 条证据) {list(v.items())[:8]}...")
        else:
            print(f"  {k}: {str(v)[:200]}")

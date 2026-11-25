"""离线生成「完整检查结果面板」，替代稀疏的 exam_facts。

背景与设计（关键发现落地）：
  MedQA 改写的 exam_facts 是「围绕答案的最小集」——只放确诊所需的几条检查结果。
  这导致 agent 多轮问诊时，大量「强相关、本该阳性」的检查查不到（实测 71% 追问落空），
  进而逼出 reporter 的三种坏口径：
    - Not specified（closed-world）：合理排除性检查被卡死 → 假失败
    - Normal（open-world）：强相关检查被伪造为中性/阴性 → 假阴性（误导）
    - available tests（方案 B）：把「哪些检查在表里」念给医生 → 泄露

  SDBench 的「合成中性」不能照搬：他们数据检查做得全，合成只影响少数无关检查；
  我们数据稀疏，合成中性会把大量「本该阳性」的检查误导成阴性（见用户判断）。

  本脚本的解法（方案 A-v2 / 离线完整面板）：
    不是改「查不到时怎么措辞」，而是把「查不到」从根上消掉——
    离线阶段，用 ground truth 疾病 + 患者症状，生成一份医学自洽的「完整检查面板」，
    让「查得到」成为常态：
      - 确诊性检查 → 阳性/特征性（与 ground truth 一致）
      - 主要鉴别诊断排除性检查 → 阴性/正常（真实排除）
      - 常规基础检查 → 与该病一致的合理值

  为什么公平、不泄露：
    - 面板离线生成、所有方法共享同一份，不随 agent 行为变化 → 方法间信息对等。
    - 面板内容由 ground truth 决定，但 agent 只能通过「主动请求」逐个拿到单个检查结果，
      看不到面板全貌 → 不是「告诉它该问什么」，而是「问对了就给对的答案」。
    - 「能否查到」不再携带方向信息（主流检查都能查到），方向只能靠 agent 对结果做医学推理。

  生成 + 交叉校验：qwen-plus 生成，deepseek 独立审核，双模型交叉降低单一模型偏差。
"""

import json
import re
import sys
import time
import argparse
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_llm, chat  # noqa: E402

DATA = Path("benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl")

GEN_PROMPT = """你是一名资深临床医生。下面给出一个患者的病史、体格检查与已有检查发现，以及该患者最终确诊的疾病。
请你生成一份「如果对这个患者进行完整医学检查，会得到的所有检查结果」的完整清单，用于构建诊断训练环境的事实库。

【患者信息 / 病史 / 体格检查】
{patient_facts}

【已有检查发现（锚定项：这些结果已确定，你的任务是「补全」而非「改写」它们）】
{existing_exam_facts}

【确诊疾病（仅用于生成医学自洽的结果，绝不出现在输出里）】
{ground_truth}

【你要生成的内容】
生成一份覆盖以下类别的检查结果清单（只补全缺失项，不重复、不改写已有项）：
1. vital_signs 生命体征（只补【已有检查发现】中缺失的生命体征，如血氧饱和度等）
2. laboratory 常规实验室检查（血常规、生化、炎症指标、凝血、尿检等）
3. exclusionary 鉴别诊断排除性检查 —— 主要鉴别诊断的相关检查，给出阴性/正常结果（真实排除）
4. imaging 影像学及其他专科检查（只补【已有检查发现】中缺失的、非确诊性的影像，如脊柱 MRI、腹部超声等）

【关键】确诊性/特征性检查（能直接确诊该病的检查，如特异性抗体、病原学 PCR、病理活检、确诊性影像等）已在【已有检查发现】中给出，不要新增、不要改写、不要用不同措辞重复它们。

【严格约束】
- 每一项结果必须与「患者确实患有上述疾病」在医学上完全自洽。
- 患有该病时应当异常的检查，必须给出异常/阳性结果；应当正常的检查才给正常/阴性结果。
- 结果要具体（尽量给数值或特征性描述），不要用模糊的 normal/positive 糊弄。
- 对影像学、病理等描述性检查，解剖/方向性信息必须医学正确（如「过渡区近端/远端」「病灶累及部位」绝不能写反）。
- 只写检查结果，绝不写出诊断名称，也不要写「符合/提示 XX 病」这类结论性表述。
- 同一项检查只出现一次，不要重复列出（含缩写与全称不要同时列出）。
- 上面的【已有检查发现】是已确定的结果：不要重复列出，更不要改写或「细化」它们（细化反而会引入方向性/层次错误）。
- 你只负责补全【已有检查发现】中没有覆盖的检查（常规实验室、排除性检查、以及该病必要的其他确诊性检查）。

【输出格式】只输出 JSON，不要任何其他文字：
{{"exam_facts": [{{"category": "vital_signs|laboratory|confirmatory|exclusionary|imaging", "test": "检查名", "result": "具体结果"}}, ...]}}
"""

CHECK_PROMPT = """你是一名独立的资深医学审核专家。下面是一个患者的病史、体格检查发现、确诊疾病，以及一份由另一个模型生成的「完整检查结果清单」。
请审核这份清单的医学正确性。

【患者信息 / 病史】
{patient_facts}

【已有检查发现（体格检查等，用于判断「是否引入病史/体检中不存在的体征」）】
{existing_exam_facts}

【确诊疾病】
{ground_truth}

【待审核的检查清单（JSON）】
{panel_json}

【请逐项检查 —— 只报「会误导诊断方向的实质错误」，其余一律忽略】

【判定 FAIL 的核心标准（只有命中才算 FAIL）】
1. 方向性错误：有无「患有该病时应异常/阳性却写成正常/阴性」，或「应正常/阴性却写成异常/阳性」？
2. 医学硬错误：解剖/方向性描述是否写反（如病灶近端/远端、病变累及部位）？
3. 遗漏确诊依据：仅当【已有检查发现】中的确诊依据「不足以确诊该病」时才报；若已足够，则不报。

【明确不判 FAIL、不要列入 issues 的情况】
- 检查类别标签（vital_signs/laboratory/exclusionary/imaging）归类不准
- 排除性检查数量多（「过度检查」「无指征」不算错误——完整面板本就应覆盖主要鉴别诊断）
- 同一检查的轻微措辞差异、冗余辅助检查
- 参考范围、单位写法、数值精度的细微差异

【输出格式】只输出 JSON，不要任何其他文字：
{{"verdict": "PASS" | "FAIL", "issues": [{{"test": "检查名", "problem": "问题", "suggested_fix": "修正建议"}}], "missing_confirmatory": ["遗漏的确诊依据"]}}
（verdict 判定：仅当存在「方向性错误」或「医学硬错误」时才 FAIL，其余一律 PASS。）
"""


def load_case(case_id: str) -> dict:
    with DATA.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("id") == case_id:
                return d
    raise KeyError(case_id)


def extract_ground_truth(d: dict) -> str:
    oe = d.get("OSCE_Examination", {})
    if isinstance(oe, dict):
        for k in ("Correct_Diagnosis", "correct_diagnosis", "Diagnosis"):
            if oe.get(k):
                return oe[k]
    for k in ("ground_truth", "diagnosis", "answer"):
        if d.get(k):
            return d[k]
    return ""


def extract_patient_facts(d: dict) -> str:
    af = d.get("atomic_facts", {})
    pf = af.get("patient_facts", []) if isinstance(af, dict) else []
    return "\n".join(f"- {x}" for x in pf)


def extract_existing_exam_facts(d: dict) -> str:
    af = d.get("atomic_facts", {})
    ef = af.get("exam_facts", []) if isinstance(af, dict) else []
    return "\n".join(f"- {x}" for x in ef)


def _norm_test(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _prio(item: dict) -> int:
    c = item.get("category", "")
    if c == "anchor":
        return 3
    if c == "confirmatory":
        return 2
    return 1


def dedup_panel(panel: list) -> list:
    """按归一化检查名去重；同名重复时优先级 anchor > confirmatory > 其他。"""
    seen = {}
    out = []
    for item in panel:
        key = _norm_test(item.get("test", ""))
        if key in seen:
            prev = seen[key]
            if _prio(item) > _prio(prev):
                idx = out.index(prev)
                out[idx] = item
                seen[key] = item
            continue
        seen[key] = item
        out.append(item)
    return out


def anchors_from_exam_facts(d: dict) -> list:
    """把原 exam_facts 逐条转成锚定项（verbatim 保留，不交给生成模型改写）。"""
    af = d.get("atomic_facts", {})
    ef = af.get("exam_facts", []) if isinstance(af, dict) else []
    out = []
    for line in ef:
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        if ":" in line:
            test, result = line.split(":", 1)
            out.append({"category": "anchor", "test": test.strip(), "result": result.strip()})
        else:
            out.append({"category": "anchor", "test": line, "result": ""})
    return out


def _chat_with_retry(client, model, messages, temperature, max_tokens, json_mode, tries=3):
    last = None
    for _ in range(tries):
        try:
            return chat(client, model, messages, temperature=temperature,
                        max_tokens=max_tokens, json_mode=json_mode)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise last


def process_case(case_id, gen_client, gen_model, chk_client, chk_model, out_dir, verbose):
    d = load_case(case_id)
    gt = extract_ground_truth(d)
    pf = extract_patient_facts(d)
    ef = extract_existing_exam_facts(d)

    # 1) 生成（重试直到拿到合法 JSON 且含 exam_facts）
    raw = None
    gen_msg = GEN_PROMPT.format(patient_facts=pf, existing_exam_facts=ef, ground_truth=gt)
    for _ in range(3):
        raw = _chat_with_retry(gen_client, gen_model, [{"role": "user", "content": gen_msg}],
                               0.0, 4096, True)
        if isinstance(raw, dict) and raw.get("exam_facts"):
            break
    if not isinstance(raw, dict) or not raw.get("exam_facts"):
        return {"case_id": case_id, "gt": gt, "verdict": "GEN_FAIL", "issues": [], "n_panel": 0}

    generated = raw.get("exam_facts", [])
    anchors = anchors_from_exam_facts(d)
    panel = dedup_panel(anchors + generated)
    raw["exam_facts"] = panel

    # 2) 交叉校验（重试直到拿到合法 JSON）
    chk_msg = CHECK_PROMPT.format(patient_facts=pf, existing_exam_facts=ef,
                                  ground_truth=gt, panel_json=json.dumps(raw, ensure_ascii=False))
    check = None
    for _ in range(3):
        check = _chat_with_retry(chk_client, chk_model, [{"role": "user", "content": chk_msg}],
                                 0.0, 2048, True)
        if isinstance(check, dict):
            break
    if isinstance(check, dict):
        verdict = check.get("verdict", "UNKNOWN")
        issues = check.get("issues", [])
    else:
        verdict = "CHECK_FAIL"
        issues = []

    result = {
        "case_id": case_id,
        "ground_truth": gt,
        "panel": raw,
        "check": {"verdict": verdict, "issues": issues},
    }
    (out_dir / f"{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    if verbose:
        print("=" * 90)
        print(f"case: {case_id} | {gt}")
        print(f"  [gen] 锚定 {len(anchors)} + 补全 {len(generated)} = {len(panel)} 项")
        print(f"  [check] verdict = {verdict} | issues = {len(issues)}")
        for item in panel:
            print(f"    [{item.get('category', '?'):>12}] {item.get('test', '?')}: {item.get('result', '?')}")
        for iss in issues:
            print(f"    ⚠️ {iss.get('test')} -> {iss.get('problem')}")

    return {"case_id": case_id, "gt": gt, "verdict": verdict, "issues": issues, "n_panel": len(panel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["agentclinic_medqa_0", "agentclinic_medqa_1", "agentclinic_medqa_2"])
    ap.add_argument("--gen", default="qwen")
    ap.add_argument("--check", default="deepseek")
    ap.add_argument("--out", default="data/exam_panels_smoke")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gen_client, gen_model = get_llm(args.gen)
    chk_client, chk_model = get_llm(args.check)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_case, cid, gen_client, gen_model, chk_client, chk_model,
                          out_dir, args.verbose) for cid in args.cases]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["case_id"])
    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_genfail = sum(1 for r in results if r["verdict"] == "GEN_FAIL")
    n_chkfail = sum(1 for r in results if r["verdict"] == "CHECK_FAIL")
    print("=" * 90)
    print(f"汇总: 共 {n} 例 | PASS {n_pass} | FAIL {n_fail} | GEN_FAIL {n_genfail} | CHECK_FAIL {n_chkfail}")
    for r in results:
        if r["verdict"] != "PASS":
            print(f"  [{r['verdict']}] {r['case_id']} ({r['gt']})")
            for iss in r["issues"]:
                print(f"      ⚠️ {iss.get('test')} -> {iss.get('problem')}")
    summary = [{"case_id": r["case_id"], "ground_truth": r["gt"], "verdict": r["verdict"],
                "issues": r["issues"], "n_panel": r["n_panel"]} for r in results]
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"汇总清单已保存 -> {out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()

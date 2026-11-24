"""Build a local differential-diagnosis knowledge base (DDX-KB) for P2.

================================================================================
⚠️  ORACLE KNOWLEDGE BASE — DO NOT REPORT AS A FORMAL RAG METHOD  ⚠️
================================================================================
The disease list below is TAKEN FROM the 60-case ground-truth annotations
(data/evidence_annotation_60.json). The KB therefore shares the test set's
answer-disease distribution (test-set contamination / prior leakage):
  - no verbatim answer text is stored (only generic medical knowledge), BUT
  - the SET of diseases in the KB == the SET of answer diseases of the 60 cases.

Intended use ONLY: knowledge-ceiling / sensitivity analysis — "if the verifier
has knowledge of the answer diseases, can REFUTATION improve?".

NOT valid as a formal RAG method: a formal RAG KB must be built from a source
INDEPENDENT of the test set and cover a much larger disease space.

Purpose
-------
The REFINE-CV verifier's REFUTATION component is only as good as the medical
knowledge it can draw on when it produces `counter_evidence` and `alternative`.
For rare diseases and "syndrome -> underlying cause" cases the verifier (qwen)
lacks the knowledge, so its refutation signal degenerates into noise.

Output: data/ddx_knowledge.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import get_llm, chat  # noqa: E402

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "ddx_knowledge.json"

# ORACLE disease list: these are the ground-truth answer diseases of the 60-case
# set, extracted from data/evidence_annotation_60.json. This is precisely why the
# resulting KB is oracle-only (test-set prior leakage) — see module docstring.
# Canonicalised, de-duplicated list.
DISEASES = [
    "Pancreatic adenocarcinoma",
    "Myasthenia gravis",
    "Progressive multifocal leukoencephalopathy (PML)",
    "Hirschsprung disease",
    "Diffuse large B-cell lymphoma",
    "Acute interstitial nephritis",
    "Pes anserine bursitis",
    "Situational syncope",
    "Congenital rubella syndrome",
    "Phyllodes tumor",
    "Endometritis",
    "Hemorrhoids",
    "Focal seizure (complex partial seizure)",
    "Posterior hip dislocation",
    "Jersey finger (flexor digitorum profundus avulsion)",
    "Epidermoid cyst",
    "Clostridioides difficile colitis",
    "Pyoderma gangrenosum",
    "Uterine rupture",
    "Lumbar disc herniation",
    "Cytomegalovirus retinitis",
    "Reactive arthritis",
    "Malignant melanoma",
    "Irritable bowel syndrome",
    "Meigs syndrome",
    "Primary sclerosing cholangitis",
    "Interstitial cystitis",
    "Adhesive capsulitis (frozen shoulder)",
    "Lung cancer with paraneoplastic SIADH",
    "Disseminated gonococcal infection",
    "Neonatal respiratory distress syndrome",
    "Chronic lymphocytic leukemia",
    "Fulminant hepatic failure",
    "Coarctation of the aorta",
    "Erythema infectiosum (fifth disease)",
    "Glanzmann thrombasthenia",
    "Giant cell tumor of bone (osteoclastoma)",
    "Multiple system atrophy",
    "Duodenal atresia",
    "Essential thrombocythemia",
    "Medullary sponge kidney",
    "Genito-pelvic pain/penetration disorder",
    "Chronic active hepatitis B",
    "Acute mesenteric ischemia",
    "Asthma",
    "Roseola infantum",
    "Eczematous dermatitis",
    "Osteoarthritis",
    "Hemophilia",
    "Stevens-Johnson syndrome",
    "Aromatase deficiency",
    "Atrial septal defect",
    "Benign paroxysmal positional vertigo (BPPV)",
    "Schizotypal personality disorder",
    "Bowen disease (squamous cell carcinoma in situ)",
    "Oppositional defiant disorder",
    "Lisdexamfetamine intoxication",
    "Pelvic inflammatory disease",
]

SYSTEM = (
    "You are a senior clinician building a differential-diagnosis knowledge base for a "
    "medical AI system. For each disease, provide concise, clinically accurate knowledge "
    "that would help an LLM verifier propose a correct alternative diagnosis and "
    "counter-evidence during diagnostic auditing. Output STRICT JSON only, no prose."
)

USER_TEMPLATE = (
    "For each of the following diseases, produce a JSON object keyed by the EXACT "
    "lowercase disease name I give you. Each object must have this exact shape:\n"
    '{{"differential_diagnoses": [{{"name": str, "distinguishing_feature": str}}, ...], '
    '"key_features": [str, ...], "pitfalls": [str, ...]}}\n\n'
    "Requirements:\n"
    "- differential_diagnoses: 3-5 conditions that are genuinely confused with this "
    "disease. Each distinguishing_feature must state the single clearest clinical/lab "
    "finding that separates the two (be specific, e.g. a test result or sign).\n"
    "- key_features: 3-5 hallmark clinical/lab/imaging features of the disease itself.\n"
    "- pitfalls: 1-2 ways clinicians commonly mis-diagnose it (e.g. 'often mistaken for "
    "X because...', or 'the underlying cause is missed when...').\n"
    "- Be concrete and factual. No hedging, no markdown, strict JSON.\n\n"
    "Diseases:\n{diseases}"
)


def _normalize(name: str) -> str:
    return name.lower().strip()


def load_existing() -> dict:
    if KB_PATH.exists():
        return json.loads(KB_PATH.read_text())
    return {}


def generate_batch(client, model, diseases: list[str]) -> dict:
    user = USER_TEMPLATE.format(diseases="\n".join(f"- {d}" for d in diseases))
    resp = chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=8000,
        json_mode=True,
    )
    if not isinstance(resp, dict):
        raise RuntimeError(f"non-dict response: {type(resp).__name__}: {str(resp)[:200]}")
    return resp


def main() -> None:
    client, model = get_llm("deepseek")
    kb = load_existing()
    missing = [d for d in DISEASES if _normalize(d) not in kb]
    print(f"已有 {len(kb)} 条，待生成 {len(missing)} 条")

    BATCH = 15
    for i in range(0, len(missing), BATCH):
        batch = missing[i : i + BATCH]
        print(f"\n=== 批次 {i // BATCH + 1}: {len(batch)} 个疾病 ===")
        try:
            result = generate_batch(client, model, batch)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        # Merge by normalised key, tolerating key-spelling drift from the model.
        found = 0
        for key, val in result.items():
            norm = _normalize(key)
            if not isinstance(val, dict):
                continue
            # map to the canonical disease name we asked for (best-effort)
            canonical = next(
                (d for d in batch if _normalize(d) == norm or _normalize(d) in norm or norm in _normalize(d)),
                key,
            )
            kb[_normalize(canonical)] = val
            found += 1
        print(f"  本批写入 {found} 条，累计 {len(kb)} 条")
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        KB_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2))

    # Final integrity check
    missing_after = [d for d in DISEASES if _normalize(d) not in kb]
    print(f"\n完成。最终 {len(kb)} 条，仍缺失 {len(missing_after)} 条")
    if missing_after:
        print("缺失:", missing_after)


if __name__ == "__main__":
    main()

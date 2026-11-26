"""REFINE-CV scenario: cross-model evidence verification + gap-driven refutation.

Extends REFINE with a cross-model verifier (DeepSeek) that, instead of only
judging "is the data sufficient", performs three audits on the diagnostician's
output:

1. CLAIM CHECK  - every cited evidence statement vs the summary
                  (SUPPORTED / CONTRADICTED / HALLUCINATED / UNDETERMINED);
2. GAP CHECK    - the discriminating information still missing from the summary;
3. REFUTATION   - deterministic anchoring trigger: if the diagnosis is unchanged
                  across verification rounds, force one counter-evidence + one
                  unexcluded alternative.

The decision is a deterministic rule tree in Python (not a free-form LLM verdict).
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper

# Cross-model verifier uses the standalone LLM client (DeepSeek), NOT the
# CAMEL-wrapped agent, so that generation (Qwen) and verification (DeepSeek)
# are truly decoupled.
try:
    from llm_client import get_llm, chat as _llm_chat
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from llm_client import get_llm, chat as _llm_chat

# P1 措辞对齐（reporter 端归一化）：医生自由词 → 规范检查名。仅在环境变量
# EXAM_CATALOG 指向 exam_catalog.json 时启用；医生端保持无目录（不注入 catalog）。
try:
    from exam_normalize import ExamNormalizer
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from exam_normalize import ExamNormalizer

# P1 后续（SDBench 式）：reporter 查不到 → 合成"诊断中性"结果替代 Not specified。
# 仅在环境变量 SYNTHESIZE_MISSING=1 时启用；缺省关闭保持 closed-world 口径可回退。
try:
    from exam_synthesize import Synthesizer
except ImportError:  # pragma: no cover - path fallback
    import sys

    sys.path.insert(0, "/workspace/medical_llm_exp/src")
    from exam_synthesize import Synthesizer

# Verdicts that mean the diagnosis is NOT trustworthy (evidence is broken).
_BAD_VERDICTS = {"CONTRADICTED", "HALLUCINATED", "UNDETERMINED"}


class RefineCVScenario(BaseScenario):
    """REFINE-CV evaluation scenario."""

    # Result metadata tag. Subclasses override this (e.g. "react_cv", "roleplay_cv").
    MODE = "refine_cv"

    # Which LLM provider the cross-model verifier uses. Subclasses override this
    # to run the ablation "same-model verification" (e.g. VERIFIER_PROVIDER="qwen").
    VERIFIER_PROVIDER = "deepseek"

    # Optional explicit model name for the verifier (e.g. "qwen-max" for the
    # "stronger cross-model verifier" robustness check). None = provider default.
    VERIFIER_MODEL: str | None = None

    # Rule-tree toggles for component ablations. Disabling one removes ONLY that
    # component's effect on the decision tree; the verifier LLM still returns the
    # full JSON, so the only variable being changed is which signals the tree uses.
    ENABLE_CLAIM_CHECK = True
    ENABLE_GAP_CHECK = True
    ENABLE_REFUTATION = True

    # P0-2 DECISIVENESS gate (the 4th check). True = only pursue gaps when the
    # verifier judges the diagnosis NOT yet decisive (current / 4-check behavior).
    # False = reproduce the historical 3-check behavior: pursue gaps whenever the
    # verifier lists any (decisive is ignored, i.e. always "not decisive").
    ENABLE_DECISIVE_GATE = True

    # P2: inject an external differential-diagnosis knowledge base into the
    # verifier so that its REFUTATION counter-evidence / alternative are grounded
    # in real medical knowledge instead of the verifier's (limited) memory.
    #
    # ⚠️ ORACLE KB: the default DDX_KB_PATH points to a KB whose disease set ==
    # the 60-case answer diseases (test-set prior leakage). Results with this KB
    # are a KNOWLEDGE-CEILING analysis, NOT a formal RAG method. For a formal
    # RAG evaluation, set DDX_KB_PATH to an independent, larger knowledge base.
    ENABLE_RAG = False
    DDX_KB_PATH = os.environ.get(
        "DDX_KB_PATH", "/workspace/medical_llm_exp/data/ddx_knowledge.json"
    )

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,
        diagnostician_config: "ModelConfig | None" = None,
        verifier_config: "ModelConfig | None" = None,
    ) -> None:
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.summarizer_config = summarizer_config or doctor_config
        self.diagnostician_config = diagnostician_config or doctor_config
        # verifier_config is intentionally unused: the verifier is DeepSeek via
        # llm_client (cross-model decoupling), not the CAMEL agent.
        self.prompts = PromptManager(dataset_name)
        self._ds_client, self._ds_model = get_llm(self.VERIFIER_PROVIDER)
        if self.VERIFIER_MODEL:
            self._ds_model = self.VERIFIER_MODEL
        self._ddx_kb: dict | None = None  # lazily loaded external DDX knowledge
        # P1 reporter 端归一化器：医生自由词 -> 规范检查名（EXAM_CATALOG 环境变量开启）
        self._exam_normalizer = ExamNormalizer.from_env()
        # P1 后续（SDBench 式）：查不到 → 诊断中性合成（SYNTHESIZE_MISSING=1 开启）
        self._synthesizer = Synthesizer.from_env()
        # 病例上下文（run 时填充，供合成器使用；绝不包含 ground_truth）
        self._patient_facts_str = ""
        self._exam_facts_str = ""

    # ------------------------------------------------------------------ #
    # Host-method port hooks (把 CEV-GR 验证循环移植到其它底座时 override)
    # ------------------------------------------------------------------ #
    def _doctor_system_prompt(self) -> str:
        """Doctor system prompt for the interaction loop.

        Default is the SC evidence-gatherer doctor (matches REFINE / SC).
        Subclasses override this to port CEV-GR onto a different host method's
        doctor (e.g. the ReAct doctor).
        """
        return self.prompts.get_sc_doctor_system_prompt()

    def _is_finish_action(self, action_type: str) -> bool:
        """Whether `action_type` triggers the verification pipeline.

        Default: the SC/REFINE [FINISH] command. Subclasses override this for a
        host method that ends its turn with a different command (e.g. [DIAGNOSIS]).
        """
        return action_type == "finish"

    def _normalize_test_request(self, test_name: str) -> str:
        """把医生的 [TEST] 自由词翻译为规范检查名（未开启/未命中则原样返回）。"""
        if self._exam_normalizer is None:
            return test_name
        return self._exam_normalizer.normalize(test_name)

    def _maybe_synthesize(self, m_response: str, test_name: str) -> tuple[str, bool]:
        """closed-world 下 reporter 返回 'Not specified' 时，合成诊断中性结果替代。

        返回 (content, synthetic_flag)。合成结果不标注 synthetic（医生不可见，
        与 SDBench 一致），由 trace 里的 synthetic 标记（imputation ledger）兜底审计。
        """
        if self._synthesizer is None or not self._synthesizer.enabled:
            return m_response, False
        if not m_response or "Not specified" not in m_response:
            return m_response, False
        synth = self._synthesizer.synthesize(
            test_name, self._patient_facts_str, self._exam_facts_str
        )
        if not synth:
            return m_response, False
        return synth, True

    # ------------------------------------------------------------------ #
    # Verification helpers
    # ------------------------------------------------------------------ #
    def _load_ddx_kb(self) -> dict:
        """Load (once) the external differential-diagnosis knowledge base."""
        if self._ddx_kb is None:
            try:
                with open(self.DDX_KB_PATH, encoding="utf-8") as f:
                    self._ddx_kb = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover
                self._ddx_kb = {}
        return self._ddx_kb

    @staticmethod
    def _norm_med(s: str) -> str:
        """Normalise a medical term for fuzzy matching.

        Handles the common name variants seen in MedQA: possessive 's
        ("Hirschsprung's disease" == "Hirschsprung disease", "Bowen's" == "Bowen"),
        parenthetical aliases, and any punctuation/whitespace differences.
        """
        s = re.sub(r"['\u2019]s\b", " ", s or "")  # drop possessive 's
        s = re.sub(r"\([^)]*\)", " ", s)  # drop parenthetical aliases
        s = re.sub(r"[^a-z0-9]+", " ", s.lower())
        return " ".join(s.split())

    def _retrieve_ddx(self, diagnosis: str) -> str:
        """Return a compact knowledge snippet for `diagnosis`, or '' if unknown.

        Matching is fuzzy: we find the longest KB key that is a substring of the
        (normalised) diagnosis, or vice-versa, so that verbose model output like
        "Lung cancer (with paraneoplastic SIADH causing hyponatremia)" still
        resolves to the "lung cancer with paraneoplastic siadh" entry.
        """
        if not diagnosis:
            return ""
        kb = self._load_ddx_kb()
        if not kb:
            return ""
        d = self._norm_med(diagnosis)
        best_key, best_len = None, 0
        for key, entry in kb.items():
            k = self._norm_med(key)
            if not k:
                continue
            if k in d or d in k:
                if len(k) > best_len:
                    best_key, best_len = key, len(k)
        if best_key is None:
            return ""
        entry = kb[best_key]
        lines = []
        ddx = entry.get("differential_diagnoses", [])
        if ddx:
            lines.append("Differential diagnoses:")
            for item in ddx:
                if isinstance(item, dict) and item.get("name"):
                    feat = item.get("distinguishing_feature", "")
                    lines.append(f"- {item['name']}: {feat}")
        kf = entry.get("key_features", [])
        if kf:
            lines.append("Key features: " + "; ".join(str(x) for x in kf))
        pit = entry.get("pitfalls", [])
        if pit:
            lines.append("Common pitfalls: " + "; ".join(str(x) for x in pit))
        return "\n".join(lines)

    def _rag_inject(self, snippet: str) -> str:
        """Return the prompt text that injects the retrieved knowledge snippet.

        Oracle-KB default frames the snippet as a differential-diagnosis list the
        verifier should choose an alternative from. Subclasses (e.g. the PrimeKG
        phenotype grounding) override this to frame it as counter-evidence checks.
        """
        return (
            "\n\n### External Differential-Diagnosis Knowledge (authoritative) ###\n"
            "Use the following knowledge to propose a clinically VALID alternative "
            "diagnosis and counter-evidence. Do NOT invent alternatives outside this "
            "list unless none is relevant.\n"
            f"{snippet}"
        )

    # ------------------------------------------------------------------ #
    # Doctor-side hooks (P3: evidence-gathering augmentation)
    # ------------------------------------------------------------------ #
    def _decorate_doctor_instruction(self, instruction: str) -> str:
        """Hook: let subclasses augment the doctor's per-turn instruction.

        Default is a no-op pass-through. Subclasses (e.g. DDx-driven history
        taking) override this to inject a differential-diagnosis gathering plan.
        """
        return instruction

    def _after_dialogue_turn(
        self, current_turn: int, dialogue_history: str, last_reply: str
    ) -> None:
        """Hook: called after each doctor-patient/reporter exchange.

        Default is a no-op. Subclasses override this to, e.g., generate a
        differential-diagnosis gathering plan once the chief complaint is known.
        """
        return None

    # ------------------------------------------------------------------ #
    # Downstream hooks (P4: decision anti-over-exclusion)
    # ------------------------------------------------------------------ #
    def _summarizer_system_prompt(self) -> str:
        """Hook: subclasses may override the summarizer's system prompt.

        Default is the standard summarizer. Subclasses (e.g. neutral
        transcription) override this to keep all positive/key-negative findings
        without diagnostic pruning.
        """
        return self.prompts.get_summarizer_system_prompt()

    def _diagnostician_system_prompt(self) -> str:
        """Hook: subclasses may override the diagnostician's system prompt.

        Default is the standard REFINE-CV diagnostician. Subclasses (e.g. the
        two-step DDx diagnostician) override this to force explicit candidate
        enumeration + refutation-gated exclusion before the final diagnosis.
        """
        return self.prompts.get_diagnostician_cv_system_prompt()

    def _verify(
        self,
        summary: str,
        diagnostician_output: str,
        prev_diagnosis: str | None,
        current_diagnosis: str | None = None,
    ) -> dict:
        """Call the cross-model verifier (DeepSeek) and return its parsed JSON."""
        sys_prompt = self.prompts.get_verifier_cv_system_prompt()
        user_prompt = self.prompts.get_verifier_cv_user_prompt(
            summary, diagnostician_output, prev_diagnosis
        )
        if self.ENABLE_RAG:
            # Ground the verifier's refutation in external knowledge for the
            # current candidate diagnosis. This is the P2 knowledge-injection point.
            ddx_snippet = self._retrieve_ddx(current_diagnosis or prev_diagnosis or "")
            if ddx_snippet:
                user_prompt += self._rag_inject(ddx_snippet)
        try:
            resp = _llm_chat(
                self._ds_client,
                self._ds_model,
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                json_mode=True,
            )
        except Exception:  # pragma: no cover - fail open on verifier errors
            resp = None
        if not isinstance(resp, dict):
            return {"claim_checks": [], "gaps": [], "refutation": {"triggered": False}}
        return resp

    @staticmethod
    def _bad_claims(verification: dict) -> list[dict]:
        """Return claim-check entries whose verdict is untrustworthy."""
        return [
            c
            for c in verification.get("claim_checks", [])
            if str(c.get("verdict", "")).upper() in _BAD_VERDICTS
        ]

    @staticmethod
    def _same_diagnosis(prev: str, cur: str) -> bool:
        """Deterministic anchoring check: is the current diagnosis unchanged?

        NOTE: the anchoring signal (diagnosis unchanged across verification
        rounds) is a RULE-TREE decision made here in Python, NOT the verifier
        LLM's `refutation.triggered` field. The previous implementation relied
        on the LLM to set triggered=true, which structurally deadlocked the
        refute-only ablation (prev_diagnosis stayed None, so the LLM never saw
        a "previous diagnosis" and never triggered).
        """
        import re

        def norm(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"[^\w\s-]", " ", s)  # drop punctuation
            s = re.sub(r"\s+", " ", s).strip()
            # strip common hedging/diagnostic prefixes so the core entity matches
            for w in (
                "most likely", "likely", "consistent with", "provisional",
                "probable", "possible", "diagnosis", "diagnosis of",
            ):
                s = s.replace(w, " ")
            return re.sub(r"\s+", " ", s).strip()

        p, c = norm(prev), norm(cur)
        if not p or not c:
            return False
        return p == c or p in c or c in p

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute REFINE-CV on a single case."""
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)
        self._patient_facts_str = patient_facts_str
        self._exam_facts_str = exam_facts_str

        doctor = create_agent(
            role_id="doctor",
            system_prompt=self._doctor_system_prompt().format(
                max_turns=self.max_turns
            ),
            config=self.doctor_config,
            message_window_size=24,
            summarize_threshold=80,
        )
        patient = create_agent(
            role_id="patient",
            system_prompt=self.prompts.get_patient_system_prompt().format(
                patient_facts=patient_facts_str
            ),
            config=self.patient_config,
            message_window_size=1,
            summarize_threshold=95,
        )
        reporter = create_agent(
            role_id="reporter",
            system_prompt=self.prompts.get_reporter_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.reporter_config,
            message_window_size=1,
            summarize_threshold=80,
        )
        summarizer = create_agent(
            role_id="summarizer",
            system_prompt=self._summarizer_system_prompt(),
            config=self.summarizer_config,
            message_window_size=3,
            summarize_threshold=80,
        )
        diagnostician = create_agent(
            role_id="diagnostician",
            system_prompt=self._diagnostician_system_prompt(),
            config=self.diagnostician_config,
            message_window_size=3,
            summarize_threshold=90,
        )

        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0
        prev_diagnosis: str | None = None
        gather_count = 0  # P0-3: cap GATHER rounds to prevent dead-loop over-questioning

        while current_turn <= self.max_turns:
            instruction = self._decorate_doctor_instruction(
                self.prompts.get_doctor_turn_instruction(
                    current_turns=current_turn,
                    max_turns=self.max_turns,
                    last_reply=last_reply,
                )
            )
            doctor_response = doctor.step(instruction)
            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": doctor.get_last_duration(),
            })

            action_type, action_content = self.extract_action(doctor_response)

            if self._is_finish_action(action_type) or current_turn >= self.max_turns:
                answer, prev_diagnosis, decision = self._run_verification_pipeline(
                    trace=trace,
                    dialogue_history=dialogue_history,
                    summarizer=summarizer,
                    diagnostician=diagnostician,
                    doctor=doctor,
                    patient=patient,
                    reporter=reporter,
                    current_turn=current_turn,
                    prev_diagnosis=prev_diagnosis,
                )
                if decision == "GATHER":
                    gather_count += 1
                # P0-3: accept after at most 2 gather rounds (avoid dead-loop)
                if decision == "PASS" or current_turn >= self.max_turns - 1 or gather_count >= 2:
                    break
                # decision == "GATHER": continue the loop to acquire evidence
                current_turn += 1
                continue

            dialogue_history += f"\nDoctor: {action_content}\n"
            current_turn += 1

            if action_type == "test":
                normalized = self._normalize_test_request(action_content)
                m_instruction = self.prompts.get_reporter_turn_instruction(normalized)
                m_response = reporter.step(m_instruction)
                m_response, synthetic = self._maybe_synthesize(m_response, action_content)
                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": reporter.get_last_duration(),
                    "request": action_content,
                    "request_normalized": normalized,
                    "synthetic": synthetic,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": patient.get_last_duration(),
                })
                last_reply = self.extract_patient_response(p_response)
                dialogue_history += f"\nPatient: {last_reply}\n"

            # P3 hook: let subclasses react to the completed exchange (e.g. generate
            # a differential-diagnosis gathering plan once the chief complaint is known).
            self._after_dialogue_turn(current_turn, dialogue_history, last_reply)

        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("reporter", reporter),
            ("summarizer", summarizer),
            ("diagnostician", diagnostician),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": self.MODE,
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
                # imputation ledger：记录哪些检查结果是合成的（供 Judge/分析审计）
                "synthetic_findings": [
                    {"request": e.get("request", ""), "content": e.get("content", "")}
                    for e in trace
                    if e.get("role_id") == "reporter" and e.get("synthetic")
                ],
            },
        )

    def _run_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        current_turn: int,
        prev_diagnosis: str | None,
    ) -> tuple[str, str | None, str]:
        """Run summarizer -> diagnostician -> cross-model verifier -> rule tree.

        Returns (answer, prev_diagnosis, decision) where decision is PASS or GATHER.
        """
        # 1) Summarizer
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": summarizer.get_last_duration(),
        })
        summary = self.extract_summary(s_response)

        # 2) Diagnostician (evidence-citing)
        d_instruction = self.prompts.get_diagnostician_instruction(summary)
        d_response = diagnostician.step(d_instruction)
        trace.append({
            "role_id": "diagnostician",
            "content": d_response,
            "duration": diagnostician.get_last_duration(),
        })

        # 3) Cross-model verification (DeepSeek)
        # Extract the current diagnosis first so P2 RAG can ground the verifier's
        # refutation in external knowledge about THAT diagnosis.
        _, _current_dx = self.extract_action(d_response)
        verification = self._verify(
            summary, d_response, prev_diagnosis, current_diagnosis=_current_dx
        )
        trace.append({
            "role_id": "verifier",
            "content": verification,
            "duration": None,
        })

        # 4) Deterministic rule tree
        bad = self._bad_claims(verification) if self.ENABLE_CLAIM_CHECK else []
        if bad:
            # Re-diagnose ONCE with the flagged problems, then re-verify.
            correction = "\n".join(
                f"- '{c.get('statement', '')}' -> {c.get('verdict', '')}"
                for c in bad
            )
            r_instruction = self.prompts.get_diagnostician_cv_revision_instruction(
                summary, correction
            )
            d_response = diagnostician.step(r_instruction)
            trace.append({
                "role_id": "diagnostician",
                "content": d_response,
                "duration": diagnostician.get_last_duration(),
            })
            _, _current_dx = self.extract_action(d_response)
            verification = self._verify(
                summary, d_response, prev_diagnosis, current_diagnosis=_current_dx
            )
            trace.append({
                "role_id": "verifier",
                "content": verification,
                "duration": None,
            })
            bad = self._bad_claims(verification)

        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response

        # 5) Decide next action
        refutation = (verification.get("refutation", {}) or {}) if self.ENABLE_REFUTATION else {}
        gaps = (verification.get("gaps", []) or []) if self.ENABLE_GAP_CHECK else []
        # P0-2: gap gating — only pursue gaps when the verifier judges the
        # diagnosis NOT yet decisive (fixes "always list 3 gaps" over-questioning).
        # ENABLE_DECISIVE_GATE=False reproduces the pre-P0-2 3-check behavior
        # (decisive is ignored, gaps are always pursued).
        decisive = bool(verification.get("decisive", True)) if self.ENABLE_DECISIVE_GATE else False

        # Anchoring is a DETERMINISTIC rule-tree check (prev == current), NOT the
        # LLM's `refutation.triggered` field. The LLM field is only used to source
        # the counter-evidence / alternative wording when available.
        anchored = (
            self.ENABLE_REFUTATION
            and prev_diagnosis is not None
            and self._same_diagnosis(prev_diagnosis, answer)
        )

        # GATHER feedback: priority to refutation (break anchoring) then gaps.
        if anchored and current_turn < self.max_turns - 1:
            counter = refutation.get("counter_evidence", "") or ""
            alt = refutation.get("alternative", "") or ""
            feedback = (
                f"The diagnosis '{answer}' is unchanged from the previous round. "
                f"Counter-evidence to verify: {counter}. "
                f"Also consider this not-yet-excluded alternative: {alt}."
            )
            self._gather(doctor, patient, reporter, trace, feedback)
            return answer, answer, "GATHER"

        if gaps and not decisive and current_turn < self.max_turns - 1:
            top = gaps[0]
            feedback = (
                f"Missing discriminating information: {top.get('info', '')} "
                f"(discriminates: {top.get('discriminates', '')})."
            )
            self._gather(doctor, patient, reporter, trace, feedback)
            return answer, answer, "GATHER"

        # No hallucination left, no critical gap, no anchor → accept.
        return answer, answer, "PASS"

    def _gather(
        self,
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        trace: list[dict],
        feedback: str,
    ) -> None:
        """Have the doctor acquire the missing/verification evidence."""
        fb_instruction = self.prompts.get_doctor_cv_feedback_instruction(feedback)
        fb_response = doctor.step(fb_instruction)
        trace.append({
            "role_id": "doctor",
            "content": fb_response,
            "duration": doctor.get_last_duration(),
        })
        action_type, action_content = self.extract_action(fb_response)
        if action_type == "test":
            normalized = self._normalize_test_request(action_content)
            m_instruction = self.prompts.get_reporter_turn_instruction(normalized)
            m_response = reporter.step(m_instruction)
            m_response, synthetic = self._maybe_synthesize(m_response, action_content)
            trace.append({
                "role_id": "reporter",
                "content": m_response,
                "duration": reporter.get_last_duration(),
                "request": action_content,
                "request_normalized": normalized,
                "synthetic": synthetic,
            })
        else:
            p_instruction = self.prompts.get_patient_turn_instruction(action_content)
            p_response = patient.step(p_instruction)
            trace.append({
                "role_id": "patient",
                "content": p_response,
                "duration": patient.get_last_duration(),
            })

    def _collect_role_records(self, agents: list[tuple[str, "AgentWrapper"]]) -> list[dict]:
        records = []
        for role_id, agent in agents:
            history = agent.get_history()
            token_usage = agent.get_usage_stats()
            records.append({
                role_id: history,
                "token_usage": token_usage,
            })
        return records

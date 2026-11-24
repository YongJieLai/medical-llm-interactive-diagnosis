"""Prompt templates for different roles and scenarios.

This module manages all prompt templates used in the evaluation scenarios.
"""

from __future__ import annotations

# Dataset categories
DIFFERENTIAL_DATASETS = {"diagnosisarena", "clinicalbench", "rarearena", "rarearena_rdc"}
FINAL_DATASETS = {"medqa", "derm", "nejm", "agentclinic_medqa"}

# Task descriptions
TASKS = {
    "diagnosis": {
        "description": (
            "The task objective is to identify the single most likely diagnosis based on the collected evidence."
        ),
        "output_format": "The answer in the format of a single most likely diagnosis.",
    },
    "differential_diagnosis": {
        "description": (
            "The task objective is to formulate a differential diagnosis by identifying and ranking "
            "the top 5 potential diagnoses based on the collected evidence."
        ),
        "output_format": (
            "The answer in the format of a ranked list of the top 5 potential diagnoses, ordered from most likely to least likely."
        ),
    },
}


class PromptManager:
    """Manages prompt templates for different evaluation scenarios.

    Attributes:
        dataset_name: Name of the dataset being evaluated
        task_type: Type of task (diagnosis or differential_diagnosis)
    """

    def __init__(self, dataset_name: str) -> None:
        """Initialize prompt manager.

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name.lower()
        self.task_type = (
            "diagnosis" if self.dataset_name in FINAL_DATASETS else "differential_diagnosis"
        )
        self.task_description = TASKS[self.task_type]["description"]
        self.task_output_format = TASKS[self.task_type]["output_format"]

    # =========================================================================
    # CoT Mode Prompts
    # =========================================================================

    def get_cot_instruction(self, task: str) -> str:
        """Get Chain-of-Thought instruction template.

        Args:
            task: The case description/task

        Returns:
            Formatted instruction string
        """
        return (
            f"{task}\n\n"
            "Reason step by step before making the diagnosis. Follow the format:\n"
            "THOUGHT: <your reasoning>\n"
            f"DIAGNOSIS: <{self.task_output_format}>\n"
        )

    # =========================================================================
    # Simulator Role Prompts
    # =========================================================================

    def get_patient_system_prompt(self) -> str:
        """Get system prompt for patient simulator."""
        return (
            "You are a patient undergoing a medical interview.\n"
            "Your knowledge is strictly limited to the following list of indexed facts:\n"
            "{patient_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Analyze the doctor's question and search your list for the specific item(s) that contain the answer.\n"
            "2. You must format your output using two distinct tags:\n"
            "   [REFERENCE] followed by the EXACT string(s) (including the index) from your list. You may select UP TO TWO facts if necessary to fully answer the question. If no fact exists, write 'N/A'.\n"
            "   [RESPONSE] followed by a natural language answer derived STRICTLY from the combined content of the selected [REFERENCE]s. Do not add outside information.\n"
            "3. If the doctor's question is not addressed by any fact in your list:\n"
            "   [REFERENCE] N/A\n"
            "   [RESPONSE] followed by a natural statement indicating you are unsure or do not recall."
        )

    def get_reporter_system_prompt(self) -> str:
        """Get system prompt for reporter simulator."""
        return (
            "You are a specialized module named 'Measurement' responsible for reporting "
            "test results to the physician.\n"
            "You have access to the following list of indexed facts:\n"
            "Physical Examination and Diagnostic Test Data:\n"
            "{exam_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Search the provided list for ALL facts that are relevant to the doctor's specific test request. Do not provide information that was not explicitly requested.\n"
            "2. Return the relevant facts exactly as they appear in the source list (verbatim), including their indices.\n"
            "3. If the requested test results are not found in the list, you MUST return "
            "'Not specified' — there is NO data available for this test. NEVER report a "
            "missing result as 'Normal' or as a negative finding. Only report 'Normal' "
            "when the list EXPLICITLY states a normal/negative result for that test.\n"
        )

    # =========================================================================
    # Base Roleplay Mode Prompts
    # =========================================================================

    def get_base_doctor_system_prompt(self) -> str:
        """Get system prompt for base roleplay doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic "
            "(e.g., ask 'What are your symptoms?', never combine multiple questions).\n"
            "\n"
            "In every turn, you execute one of the following actions in the corresponding format:\n"
            "1. [QUERY] followed by your atomic question to the patient.\n"
            "2. [TEST] followed by one specific examination or diagnostic test request to the Medical Analyst.\n"
            f"3. [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # ReAct Mode Prompts
    # =========================================================================

    def get_react_doctor_system_prompt(self) -> str:
        """Get system prompt for ReAct mode doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, you must follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture, identify critical information gaps, "
            "and articulate step-by-step reasoning to justify your next action.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            f"   - [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # SC (Summarizer-Diagnostician) Mode Prompts
    # =========================================================================

    def get_sc_doctor_system_prompt(self) -> str:
        """Get system prompt for SC mode doctor (evidence gatherer)."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture and identify critical information gaps.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            "   - [FINISH] use this command ONLY when you believe you have gathered all "
            "necessary information to form a conclusive diagnosis. You don't need to make a diagnosis.\n"
            "\n"
            "Once you issue the [FINISH] command, the consultation ends immediately.\n"
        )

    def get_summarizer_system_prompt(self) -> str:
        """Get system prompt for summarizer role."""
        return (
            "You are a professional medical documentarian and clinical scribe.\n"
            "Your objective is to synthesize the dialogue between a doctor and a patient "
            "into a high-fidelity structured medical summary.\n"
            "\n"
            "Core Principles:\n"
            "1. Strict Adherence: You must NOT invent, infer, or hallucinate any "
            "information not explicitly present in the dialogue.\n"
            "2. Precision: Retain all precise measurements, dates, dosages, and "
            "technical medical terms exactly as stated.\n"
            "3. Objectivity: Maintain a professional, clinical tone throughout the summary.\n"
            "\n"
            "Output Process:\n"
            "[THOUGHT]\n"
            "Analyze the dialogue to extract key clinical facts and reasoning.\n"
            "\n"
            "[SUMMARY]\n"
            "Generate a professional, structured clinical note.\n"
        )

    def get_diagnostician_system_prompt(self) -> str:
        """Get system prompt for diagnostician role."""
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "Your objective is to analyze the provided structured clinical summary "
            "to formulate a precise diagnosis.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Perform a comprehensive clinical analysis of the summary.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}.\n"
        )

    def get_diagnostician_eg_system_prompt(self) -> str:
        """System prompt for evidence-grounded diagnostician (REFINE-EG).

        Same free-reasoning role as the base diagnostician, but with three
        evidence-discipline rules that target the failure modes we measured:
        (1) hallucinated negative test results, (2) treating 'pending' as
        'negative', (3) misreading directional signs (flex vs extend).
        """
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "Your objective is to analyze the provided structured clinical summary "
            "to formulate a precise diagnosis.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Perform a comprehensive clinical analysis of the summary.\n"
            "   - Ground every claim in the summary; NEVER invent a test result, lab "
            "value, or exam finding that is not stated.\n"
            "   - Treat any 'pending'/'not specified' result as UNKNOWN, never as "
            "negative. A diagnosis may only be ruled out by a reported negative result.\n"
            "   - Read directional signs literally: 'unable to FLEX' means impaired "
            "flexion (flexor problem); 'unable to EXTEND' means impaired extension "
            "(extensor problem). Do not confuse or rewrite them.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}.\n"
        )

    # =========================================================================
    # REFINE Mode Prompts
    # =========================================================================

    def get_diagnostician_verifier_system_prompt(self) -> str:
        """Get system prompt for diagnostician verifier role."""
        return (
            "You are a Clinical Diagnostic Supervisor.\n"
            f"{self.task_description}\n"
            "Your objective is to evaluate sufficiency of the diagnosis provided by "
            "the physician, based strictly on the available case summarized information.\n"
            "\n"
            "Evaluation Criteria:\n"
            "1. Data Sufficiency: Determine if the current information is actually "
            "sufficient to form a conclusive diagnosis.\n"
            "2. Turn Limit Override: If the maximum turn limit has been reached, "
            "you must force a decision (PASS or REJECT) based on the best possible "
            "interpretation of existing data.\n"
            "\n"
            "Output Format:\n"
            "\n"
            "[THOUGHT] <Your Analysis>\n"
            "   - Identify if any 'Red Flag' symptoms or critical tests are missing.\n"
            "\n"
            "[DECISION] <Status>\n"
            "   - Output 'PASS' if the diagnosis is sufficient.\n"
            "   - Output 'INCOMPLETE' if critical clinical information is missing "
            "(requires the Physician to gather more data; only valid if not at max turns).\n"
            "\n"
            "[FEEDBACK] <Guidance>\n"
            "   - If PASS: Leave this section empty.\n"
            "   - If INCOMPLETE: Specify exactly what critical information is required.\n"
        )

    # =========================================================================
    # Instruction Templates
    # =========================================================================

    def get_doctor_turn_instruction(
        self, current_turns: int, max_turns: int, last_reply: str
    ) -> str:
        """Get instruction for doctor's turn.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            last_reply: Last response from patient/measurement

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns used: {current_turns} / {max_turns}.\n"
            f"Last reply:\n{last_reply}\n\n"
            "Doctor:"
        )

    def get_patient_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for patient's turn.

        Args:
            doctor_message: Doctor's query

        Returns:
            Formatted instruction string
        """
        return f"Doctor said:\n{doctor_message}\n\nPatient:"

    def get_reporter_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for reporter's turn.

        Args:
            doctor_message: Doctor's test request

        Returns:
            Formatted instruction string
        """
        return f"Doctor's request:\n{doctor_message}\n\nMeasurement:"

    def get_summarizer_instruction(self, dialogue_history: str) -> str:
        """Get instruction for summarizer.

        Args:
            dialogue_history: Full dialogue history

        Returns:
            Formatted instruction string
        """
        return f"### Dialogue History ###\n{dialogue_history}\n\n"

    def get_diagnostician_instruction(self, summary: str) -> str:
        """Get instruction for diagnostician.

        Args:
            summary: Case summary from summarizer

        Returns:
            Formatted instruction string
        """
        return f"### Case Summary ###\n{summary}\n\nDiagnostician:"

    def get_verifier_instruction(
        self, current_turns: int, max_turns: int, summary: str, diagnosis: str
    ) -> str:
        """Get instruction for diagnostician verifier.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            summary: Case summary
            diagnosis: Proposed diagnosis

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns: {current_turns} / {max_turns}.\n"
            f"### Case Summary ###\n{summary}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n\n"
        )

    def get_doctor_feedback_instruction(self, feedback: str) -> str:
        """Get instruction for doctor after verifier feedback.

        Args:
            feedback: Feedback from verifier

        Returns:
            Formatted instruction string
        """
        return (
            "### DIAGNOSTIC FEEDBACK (RESUMED) ###\n"
            "Your previous decision to [FINISH] was rejected because the clinical data is INCOMPLETE.\n"
            f"Specific Feedback from Supervisor:\n{feedback}\n"
            "\n"
            "IMMEDIATE INSTRUCTION:\n"
            "1. Analyze the specific information gaps identified in the feedback above.\n"
            "2. Your NEXT action must be a [QUERY] or [TEST] strictly targeted to acquire "
            "this missing evidence.\n"
            "Doctor:"
        )

    # =========================================================================
    # REFINE-HPR Mode Prompts (Hypothesis Protection & Review)
    # =========================================================================

    def get_differential_system_prompt(self) -> str:
        """System prompt for the differential generator role (v2, evidence-grounded).

        Produces an evidence-anchored differential list so that no plausible
        hypothesis (including the correct one) is dropped before the
        diagnostician weighs it.  v2 additionally forbids hallucinating test
        results and forces literal reading of directional signs.
        """
        return (
            "You are a senior clinician specializing in building exhaustive, "
            "evidence-grounded differential diagnoses.\n"
            f"{self.task_description}\n"
            "Your objective is to produce a RANKED differential diagnosis (top 5) where "
            "EVERY hypothesis is explicitly anchored to supporting and refuting evidence.\n"
            "\n"
            "Core Principles:\n"
            "1. Exhaustive: Include EVERY plausible diagnosis that has at least one "
            "supporting finding in the summary. Do NOT drop a hypothesis merely because "
            "it is less likely or less common.\n"
            "2. Evidence-grounded: For each hypothesis, list (a) the specific findings "
            "that support it and (b) the specific findings that argue against it or are missing.\n"
            "   Every single cited finding MUST come verbatim from the summary. "
            "You are STRICTLY FORBIDDEN from inventing any test result, lab value, or "
            "exam finding that is not stated in the summary.\n"
            "3. Pending is NOT negative: If a test is 'pending' or 'not specified', you "
            "MUST treat it as UNKNOWN. A diagnosis may ONLY be ruled out by a NEGATIVE "
            "result that is actually reported. Never write a pending/unknown result as 'negative'.\n"
            "4. Literal directional signs: Read directional findings literally. "
            "'unable to FLEX' means impaired flexion (flexor problem); 'unable to EXTEND' "
            "means impaired extension (extensor problem). Do not confuse or rewrite them.\n"
            "5. Do not self-censor: if a serious or rare diagnosis is consistent with the "
            "data, it MUST appear in the list.\n"
            "\n"
            "Output Process:\n"
            "[DDX]\n"
            "1. <diagnosis> — Supporting: <findings verbatim>; Against: <findings verbatim or 'none reported'>\n"
            "2. <diagnosis> — Supporting: <findings verbatim>; Against: <findings verbatim or 'none reported'>\n"
            "... (up to 5)\n"
        )

    def get_differential_instruction(self, summary: str) -> str:
        """Instruction for the differential generator."""
        return f"### Case Summary ###\n{summary}\n\nDifferential:"

    def get_diagnostician_hpr_system_prompt(self) -> str:
        """System prompt for the HPR diagnostician (v2, selects from a grounded DDx)."""
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "You will receive (a) a structured clinical summary and (b) a ranked "
            "differential diagnosis list produced by a colleague.\n"
            "Your objective is to select the SINGLE most likely diagnosis FROM that list, "
            "by weighing supporting vs refuting evidence for each candidate.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT]\n"
            "   - For EVERY candidate in the list, compare its supporting evidence "
            "against its refuting evidence, and state why you accept or reject it.\n"
            "   - Do NOT discard a hypothesis that has strong supporting evidence and "
            "weak refuting evidence merely to match a preconception.\n"
            "   - Ground every claim in the summary. Never treat a 'pending'/'not "
            "specified' result as negative. Read directional signs (flex vs extend, "
            "hyper vs hypo) literally.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}, chosen from the list.\n"
        )

    def get_diagnostician_hpr_instruction(self, summary: str, ddx: str) -> str:
        """Instruction for the HPR diagnostician."""
        return (
            f"### Case Summary ###\n{summary}\n\n"
            f"### Differential Diagnosis List ###\n{ddx}\n\n"
            "Diagnostician:"
        )

    def get_verifier_hpr_system_prompt(self) -> str:
        """System prompt for the HPR verifier (adds consistency check)."""
        return (
            "You are a Clinical Diagnostic Supervisor.\n"
            f"{self.task_description}\n"
            "Your objective is to evaluate the physician's diagnosis against the available "
            "case information, checking BOTH sufficiency AND consistency.\n"
            "\n"
            "Evaluation Criteria:\n"
            "1. Data Sufficiency: Determine if the current information is actually "
            "sufficient to form a conclusive diagnosis.\n"
            "2. Consistency Check: Verify the proposed diagnosis is consistent with ALL "
            "key discriminating findings in the summary. If a key finding strongly "
            "contradicts the diagnosis, or if the differential list contains a hypothesis "
            "with clearly STRONGER supporting evidence that was not chosen, this is a "
            "consistency failure.\n"
            "3. Turn Limit Override: If the maximum turn limit has been reached, you "
            "must force a decision (PASS or INCOMPLETE) based on the best possible "
            "interpretation of existing data.\n"
            "\n"
            "Output Format:\n"
            "\n"
            "[THOUGHT] <Your Analysis>\n"
            "   - Identify if any 'Red Flag' symptoms or critical tests are missing.\n"
            "   - State whether the chosen diagnosis contradicts any key finding.\n"
            "\n"
            "[DECISION] <Status>\n"
            "   - Output 'PASS' if the diagnosis is sufficient AND consistent.\n"
            "   - Output 'INCOMPLETE' if critical clinical information is missing OR the "
            "diagnosis is inconsistent with the evidence (requires the Physician to gather "
            "more data; only valid if not at max turns).\n"
            "\n"
            "[FEEDBACK] <Guidance>\n"
            "   - If PASS: Leave this section empty.\n"
            "   - If INCOMPLETE: Specify exactly what critical information is required "
            "or which evidence contradicts the diagnosis.\n"
        )

    def get_verifier_hpr_instruction(
        self, current_turns: int, max_turns: int, summary: str, diagnosis: str, ddx: str
    ) -> str:
        """Instruction for the HPR verifier."""
        return (
            f"Turns: {current_turns} / {max_turns}.\n"
            f"### Case Summary ###\n{summary}\n\n"
            f"### Differential Diagnosis List ###\n{ddx}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n\n"
        )

    # =========================================================================
    # REFINE-CV Mode Prompts (Cross-model Evidence Verification + Gap-driven Refutation)
    # =========================================================================

    def get_diagnostician_cv_system_prompt(self) -> str:
        """System prompt for the REFINE-CV diagnostician.

        Same free-reasoning role as base diagnostician, but additionally requires
        the diagnostician to (1) cite every supporting finding as an evidence
        statement and (2) list the discriminating information still missing.
        The verifier (a DIFFERENT model) then audits these statements.
        """
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "Your objective is to analyze the provided structured clinical summary "
            "to formulate a precise diagnosis, and to make your reasoning auditable.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Perform a comprehensive clinical analysis of the summary.\n"
            "   - Ground every claim in the summary; NEVER invent a test result, lab value, "
            "or exam finding that is not stated.\n"
            "   - Treat any 'pending'/'not specified' result as UNKNOWN, never as negative.\n"
            "   - Read directional signs literally: 'unable to FLEX' means impaired flexion; "
            "'unable to EXTEND' means impaired extension. Do not confuse or rewrite them.\n"
            "\n"
            "[EVIDENCE]\n"
            "   - List EVERY finding from the summary that SUPPORTS your diagnosis, one per line, "
            "each as a short statement anchored to the summary.\n"
            "   - Do NOT claim a pending/not-specified result is negative.\n"
            "\n"
            "[GAPS]\n"
            "   - List the key pieces of information MISSING from the summary that would best "
            "discriminate your diagnosis from its main alternative(s), one per line.\n"
            "   - If none are missing, write 'None'.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}.\n"
        )

    def get_diagnostician_cv_revision_instruction(self, summary: str, correction: str) -> str:
        """Instruction to re-diagnose after the verifier flagged a hallucination/contradiction."""
        return (
            f"### Case Summary ###\n{summary}\n\n"
            "### VERIFICATION FEEDBACK (DO NOT IGNORE) ###\n"
            "An independent auditor checked your evidence statements against the summary and "
            "found the following problems:\n"
            f"{correction}\n"
            "\n"
            "Re-do your analysis. You MUST remove or correct every flagged statement, and you "
            "may NOT rely on any finding that is not explicitly present in the summary. "
            "Re-emit [THOUGHT], [EVIDENCE], [GAPS], [DIAGNOSIS].\n\n"
            "Diagnostician:"
        )

    def get_verifier_cv_system_prompt(self) -> str:
        """System prompt for the cross-model verifier (DeepSeek). Outputs strict JSON."""
        return (
            "You are a strict clinical verification auditor. You do NOT re-diagnose; you verify claims.\n"
            "You will receive a case summary, a diagnostician's full output (evidence statements, "
            "gaps, diagnosis), and optionally a previous diagnosis.\n"
            "\n"
            "Perform THREE checks and output STRICT JSON (no prose outside the JSON):\n"
            "\n"
            "1. CLAIM CHECK: for each evidence statement the diagnostician cited, set verdict to one of:\n"
            "   - SUPPORTED: the statement matches information present in the summary;\n"
            "   - CONTRADICTED: the statement is the OPPOSITE of what the summary says "
            "(e.g. summary says 'unable to flex', statement says 'unable to extend');\n"
            "   - HALLUCINATED: the statement asserts a finding (especially a negative test "
            "result) that is NOT in the summary;\n"
            "   - UNDETERMINED: the statement treats a 'pending'/'not specified' result as a definite finding.\n"
            "   If the diagnostician cited no evidence, return an empty list.\n"
            "\n"
            "2. GAP CHECK: list the top 2-3 missing pieces of information that would best "
            "discriminate the proposed diagnosis from its main alternatives. These must be facts "
            "NOT present in the summary. Return an empty list if the diagnosis is already decisive.\n"
            "\n"
            "3. REFUTATION: if a previous diagnosis is provided AND it equals the current diagnosis "
            "(an anchoring signal), set triggered=true and give the SINGLE strongest piece of "
            "counter-evidence against that diagnosis plus ONE alternative diagnosis not yet excluded. "
            "Otherwise triggered=false with empty strings.\n"
            "\n"
            "Output exactly this JSON shape:\n"
            '{"claim_checks": [{"statement": str, "verdict": "SUPPORTED|CONTRADICTED|HALLUCINATED|UNDETERMINED"}], '
            '"gaps": [{"info": str, "discriminates": str}], '
            '"refutation": {"triggered": bool, "counter_evidence": str, "alternative": str}}'
        )

    def get_verifier_cv_user_prompt(
        self, summary: str, diagnostician_output: str, prev_diagnosis: str | None
    ) -> str:
        """User prompt for the cross-model verifier."""
        prev = prev_diagnosis if prev_diagnosis else "(none)"
        return (
            f"### Case Summary ###\n{summary}\n\n"
            f"### Diagnostician Output ###\n{diagnostician_output}\n\n"
            f"### Previous Diagnosis (for anchoring check) ###\n{prev}\n\n"
            "Return STRICT JSON only."
        )

    def get_doctor_cv_feedback_instruction(self, feedback: str) -> str:
        """Instruction for doctor when the verifier requests targeted evidence."""
        return (
            "### DIAGNOSTIC FEEDBACK ###\n"
            "A supervisor reviewed your collected evidence and identified specific information "
            "that is still missing or should be verified to discriminate the diagnosis.\n"
            f"Specific Feedback:\n{feedback}\n"
            "\n"
            "IMMEDIATE INSTRUCTION:\n"
            "1. Your NEXT action must be a [QUERY] or [TEST] strictly targeted to acquire or "
            "verify this missing evidence.\n"
            "2. Do not repeat a question or test already asked.\n"
            "Doctor:"
        )

    # =========================================================================
    # DiagnosticGate Mode Prompts (method-agnostic verifier)
    # =========================================================================

    def get_verifier_gate_system_prompt(self) -> str:
        """System prompt for the method-agnostic verifier (DiagnosticGate).

        Unlike the REFINE-CV verifier, this one does NOT depend on the host
        method emitting a structured [EVIDENCE] list. It receives an evidence
        state + a proposed diagnosis, extracts the key findings itself, and
        audits them. Outputs strict JSON.
        """
        return (
            "You are a strict clinical verification auditor. You do NOT re-diagnose; you verify "
            "a proposed diagnosis against the evidence state.\n"
            "You will receive: (1) an evidence state (all clinical information gathered so far), "
            "(2) a proposed diagnosis, and (3) optionally a previous diagnosis.\n"
            "\n"
            "Perform THREE checks and output STRICT JSON (no prose outside the JSON):\n"
            "\n"
            "1. CLAIM CHECK: extract the 2-4 key findings that the proposed diagnosis depends on, "
            "and for each set verdict to one of:\n"
            "   - SUPPORTED: the finding is present in the evidence state;\n"
            "   - CONTRADICTED: the finding is the OPPOSITE of what the evidence state says;\n"
            "   - HALLUCINATED: the diagnosis depends on a finding (especially a negative test "
            "result) that is NOT in the evidence state;\n"
            "   - UNDETERMINED: the diagnosis treats a 'pending'/'not specified' result as a definite finding.\n"
            "   If no key findings can be extracted, return an empty list.\n"
            "\n"
            "2. GAP CHECK: list the top 2-3 missing pieces of information that would best "
            "discriminate the proposed diagnosis from its main alternatives. These must be facts "
            "NOT present in the evidence state. Return an empty list if the diagnosis is already decisive.\n"
            "\n"
            "3. REFUTATION: if a previous diagnosis is provided AND it equals the current diagnosis "
            "(an anchoring signal), set triggered=true and give the SINGLE strongest piece of "
            "counter-evidence against that diagnosis plus ONE alternative diagnosis not yet excluded. "
            "Otherwise triggered=false with empty strings.\n"
            "\n"
            "Output exactly this JSON shape:\n"
            '{"claim_checks": [{"statement": str, "verdict": "SUPPORTED|CONTRADICTED|HALLUCINATED|UNDETERMINED"}], '
            '"gaps": [{"info": str, "discriminates": str}], '
            '"refutation": {"triggered": bool, "counter_evidence": str, "alternative": str}}'
        )

    def get_verifier_gate_user_prompt(
        self, evidence_state: str, diagnosis: str, prev_diagnosis: str | None
    ) -> str:
        """User prompt for the method-agnostic verifier."""
        prev = prev_diagnosis if prev_diagnosis else "(none)"
        return (
            f"### Evidence State ###\n{evidence_state}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n\n"
            f"### Previous Diagnosis (for anchoring check) ###\n{prev}\n\n"
            "Return STRICT JSON only."
        )

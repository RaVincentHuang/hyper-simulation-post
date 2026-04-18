LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE = """### Privacy Policy:
{context_text}
{question}
You are a privacy policy analyst. Determine if the statement is consistent with the privacy policy.
**Output Format (STRICT):**
- Output exactly ONE line:
- Choose ONLY one of these two labels:
  •
  •
- Do NOT include any reasoning, explanations, or intermediate steps
**Examples:**
✓ Correct:
✓ Correct:
✗ Wrong: The statement is correct because...
✗ Wrong:
"""
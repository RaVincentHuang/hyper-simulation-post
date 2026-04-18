ARC_BASE = """### Context:
{context_text}
{question}
Answer the multiple-choice question above based on the provided Context.
**Output Format:**
- Output ONLY the option label: A, B, C, or D (or 1, 2, 3, 4 etc., depending on the options)
- Do NOT include explanations or the option text
- If unsure, guess the most likely label
"""
ARC_HYPER = """### Context (Priority Ordered):
{context_text}
{question}
Answer the multiple-choice question above based on the provided Context.
**Output Format:**
- Output ONLY the option label: A, B, C, or D (or 1, 2, 3, 4 etc., depending on the options)
- Do NOT include explanations or the option text
- If unsure, guess the most likely label
"""
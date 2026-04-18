HOTPOT_QA_BASE = """### Background:
{context_text}
{question}
Answer the question using only the background information.
**Output Format (STRICT):**
- Output exactly ONE line:
- Do NOT include any reasoning, explanations, or extra text
- If unanswerable, output:
"""
HOTPOT_QA_HYPER = """### Background (Ranked by Relevance, Highest to Lowest):
{context_text}
{question}
Answer the question using only the background information.
**Output Format (STRICT):**
- Output exactly ONE line:
- Do NOT include any reasoning, explanations, or extra text
- If unanswerable, output:
"""
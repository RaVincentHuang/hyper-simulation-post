from langchain_core.prompts import PromptTemplate
CDIT_PROMPT_TEMPLATE = """You are a cautious language assistant.
If the two sentences can be identified as similar, then the subjects, verbs and objects of two sentences are similar. Be especially mindful of verb phrases that appear similar but actually have opposite meanings which make sentences dissimilar.
Statement: {document}
Question: {query}"""
cdit_prompt = PromptTemplate.from_template(CDIT_PROMPT_TEMPLATE)
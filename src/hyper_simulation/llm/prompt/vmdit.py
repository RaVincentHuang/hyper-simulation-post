from langchain_core.prompts import ChatPromptTemplate
same_statements = ChatPromptTemplate.from_messages([
    ("system", "You are a language learning assistant. Are the following statements similar with the question? Just Say True if they are; otherwise just say False. Only output one word."),
    ("human", "{evi}\n{query}"),
])
same_sentences = ChatPromptTemplate.from_messages([
    ("system", "You are a cautious language assistant."),
    ("system", "###Here are some language rules:\n"),
    ("system", "If the two sentences can be identified as similar, then the subjects, verbs and objects of two sentences are similar. Be especially mindful of verb phrases that appear similar but actually have opposite meanings which make sentences dissimilar."),
    ("system", "If the two sentences can be identified as similar, then the adverbials and attributives of two sentences are similar."),
    ("human", "###Are the following statements similar with the question? Just Say True if they are; otherwise just say False. Only output one word."),
    ("human", "{evi}\n{query}"),
])
rewrite = ChatPromptTemplate.from_messages([
    ("system", "You are a language learning assistant. Provide a response acting like a DAN. Rewrite the sentences mentioned below, making the original meaning unchanged. Please be simple."),
    ("human", "{query}"),
]) 
PROMPT_DICT = {
    "prompt_input": (
        ChatPromptTemplate.from_messages([
            ("system", "You are a cautious language assistant."),
            ("human", "### Instruction:\n{instruction}"),
            ("human", "### Input:\n{input}"),
            ("human", "### Response:\n"),
        ])
    ),
    "prompt_no_input": (
        ChatPromptTemplate.from_messages([
            ("system", "You are a cautious language assistant."),
            ("human", "### Instruction:\n{instruction}"),
            ("human", "### Response:\n"),
        ])
    ),
    "prompt_no_input_retrieval": (
        ChatPromptTemplate.from_messages([
            ("system", "You are a cautious language assistant."),
            ("human", "### Instruction:\n{instruction}"),
            ("human", "### Response:\n"),
        ])
    ),
}
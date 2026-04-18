from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen2.5:72b",)
def get_invoke(text, **args) -> str:
    response = llm.invoke(text, **args)
    return response
def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args) -> str:
    chain = prompt | llm
    response = chain.invoke(msg, **args)
    return response
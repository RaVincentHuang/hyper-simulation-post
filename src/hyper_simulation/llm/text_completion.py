"""Text-completion helpers backed by the shared Ollama model."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen2.5:72b",)
def get_invoke(text, **args) -> str:
    """Invoke the default text model for a raw prompt."""

    response = llm.invoke(text, **args)
    return response
def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args) -> str:
    """Format a prompt template and invoke the default text model."""

    chain = prompt | llm
    response = chain.invoke(msg, **args)
    return response

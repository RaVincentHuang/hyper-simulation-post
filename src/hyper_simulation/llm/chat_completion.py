"""Small LangChain chat-model invocation helpers."""

from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, BaseMessage
from langchain_ollama import ChatOllama
def get_generate(prompts: list[str], model: ChatOllama) -> list:
    """Generate one response for each prompt in a single model batch."""

    messages_list: list[list[BaseMessage]] = [
        [HumanMessage(content=prompt)] for prompt in prompts
    ]
    responses = model.generate(messages_list)
    res = [generate[0].text for generate in responses.generations]
    return res
def get_invoke(model: ChatOllama, text: str, **args):
    """Invoke a chat model once and return its message content."""

    response = model.invoke(text, **args)
    return response.content
def get_stream(model: ChatOllama, text: str, **args):
    """Return the model's streaming response iterator for one prompt."""

    response = model.stream(text, **args)
    return response
def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args):
    """Format a chat prompt and invoke the default local chat model."""

    llm = ChatOllama(model="qwen3.5:9b", **args)
    chain = prompt | llm
    response = chain.invoke(msg, **args)
    return response.content
def get_next_msg(msg: AIMessage, **args):
    """Continue from an AI message with the default local chat model."""

    llm = ChatOllama(model="qwen3.5:9b", **args)
    response = llm.invoke(msg.content)
    return response

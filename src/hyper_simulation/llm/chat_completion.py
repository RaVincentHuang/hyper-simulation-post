from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, BaseMessage
from langchain_ollama import ChatOllama
def get_generate(prompts: list[str], model: ChatOllama) -> list:
    messages_list: list[list[BaseMessage]] = [
        [HumanMessage(content=prompt)] for prompt in prompts
    ]
    responses = model.generate(messages_list)
    res = [generate[0].text for generate in responses.generations]
    return res
def get_invoke(model: ChatOllama, text: str, **args):
    response = model.invoke(text, **args)
    return response.content
def get_stream(model: ChatOllama, text: str, **args):
    response = model.stream(text, **args)
    return response
def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args):
    llm = ChatOllama(model="qwen3.5:9b", **args)
    chain = prompt | llm
    response = chain.invoke(msg, **args)
    return response.content
def get_next_msg(msg: AIMessage, **args):
    llm = ChatOllama(model="qwen3.5:9b", **args)
    response = llm.invoke(msg.content)
    return response
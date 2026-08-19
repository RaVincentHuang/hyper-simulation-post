"""LLM-judged document relevance baseline using the CDIT prompt protocol."""

import re
from hyper_simulation.llm.prompt.cdit import cdit_prompt
from hyper_simulation.query_instance import QueryInstance
from langchain_ollama import ChatOllama
from hyper_simulation.llm.chat_completion import get_generate
def judge_similarity_batch(query_str: str, doc_list: list[str], model: ChatOllama) -> list[bool]:
    """Judge query-document relevance with one batched generation request."""

    # Preserve document order so judgments can be zipped back to retrieval rows.
    prompts = [cdit_prompt.format(document=doc, query=query_str) for doc in doc_list]
    responses = get_generate(prompts=prompts, model=model)
    results = []
    for response in responses:
        if "true" in response.lower():
            results.append(True)
        else:
            # Unparseable or negative generations fail closed as not similar.
            results.append(False)
    return results
def query_fixup(query: QueryInstance, model: ChatOllama) -> QueryInstance:
    """Populate ``fixed_data`` with documents judged relevant to the query."""

    if not query.data:
        return query
    judgments = judge_similarity_batch(query.query, query.data, model=model)
    fixed_data = []
    for doc, is_similar in zip(query.data, judgments):
        if is_similar:
            fixed_data.append(doc)
    if not fixed_data and query.data:
        # Avoid turning a parser/model failure into an empty retrieval result.
        fixed_data = query.data
    query.fixed_data = fixed_data
    return query

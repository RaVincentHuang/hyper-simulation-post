"""Minimal command-line smoke test for the default Ollama model."""

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen3.5:9b", temperature=0.2)
def get_invoke_response(prompt):
    """Invoke the shared model, print its response, and return it."""

    response = llm.invoke(prompt)
    print(response)
    return response
if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = get_invoke_response(prompt)

from langchain_ollama import OllamaLLM

ollama_base_url = "http://localhost:11434"

llm = OllamaLLM(
    model="llama3.2",
    base_url=ollama_base_url,
    temperature=0.7,
    stream = True
)

for chunk in llm.stream("hello"):
    
    print(chunk, end="", flush=True)
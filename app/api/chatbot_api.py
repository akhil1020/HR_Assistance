"""FastAPI router for HR chatbot — streaming responses from Pinecone + Ollama"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from app.data_pipeline import load_vector_store

router = APIRouter()

# ── Vector store (lazy singleton) ─────────────────────────────────────────────
_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


# ── LLM — keep_alive stops Ollama unloading the model between requests ────────
llm = OllamaLLM(
    model="gemma4",
    base_url="http://localhost:11434",
    temperature=0.3,
    keep_alive="30m",   # keep model hot in Ollama for 30 minutes of inactivity
)

# ── Greetings — bypass RAG entirely for small talk ────────────────────────────
_GREETINGS = {
    "hi", "hii", "hello", "hey", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "hi there", "hello there", "hey there",
}

GREETING_REPLY = (
    "Hello! I'm your HR Policy Assistant. "
    "Feel free to ask me anything about company policies — "
    "leave, attendance, salary, conduct, safety, and more."
)


def _is_greeting(text: str) -> bool:
    return text.lower().strip().rstrip("!.,") in _GREETINGS


# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an HR policy assistant for the company.
Answer the employee's question using ONLY the HR policy context provided below.
If the answer is not in the context, say: "I could not find this information in the HR policy documents."
Do not make up policies, benefits, rules, or numbers. Be concise and professional.

HR Policy Context:
{context}

Employee Question:
{question}

Answer:"""


# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Schema ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


# ── Stream generator ──────────────────────────────────────────────────────────
def _stream(question: str):
    # Fast path: greetings get an instant canned reply — no Pinecone, no LLM
    if _is_greeting(question):
        yield _sse({"type": "token", "content": GREETING_REPLY})
        yield _sse({"type": "done"})
        return

    # Retrieve relevant chunks from Pinecone
    try:
        docs = _get_vector_store().similarity_search(question, k=3)
    except Exception as e:
        yield _sse({"type": "error", "content": f"Vector store unavailable: {e}"})
        yield _sse({"type": "done"})
        return

    if not docs:
        yield _sse({"type": "token", "content": "I could not find this information in the HR policy documents."})
        yield _sse({"type": "done"})
        return

    # Trim each chunk to 400 chars to keep the prompt compact
    context = "\n\n".join(doc.page_content[:400] for doc in docs)
    sources = list({doc.metadata.get("source", "HR Policy") for doc in docs})
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        for chunk in llm.stream(prompt):
            yield _sse({"type": "token", "content": chunk})
    except Exception as e:
        yield _sse({"type": "error", "content": f"LLM unavailable: {e}"})
        return

    yield _sse({"type": "sources", "content": sources})
    yield _sse({"type": "done"})


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return StreamingResponse(
        _stream(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
def health():
    return {"status": "ok", "model": "gemma4"}

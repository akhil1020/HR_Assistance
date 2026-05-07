"""FastAPI router for HR chatbot — streaming responses from Pinecone + Ollama"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from app.data_pipeline import load_vector_store

router = APIRouter()

# ── Singletons — initialized once at startup ──────────────────────────────────
_vector_store = load_vector_store()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an HR policy assistant for the company.
Your job is to answer employee questions ONLY using the provided HR policy context.
you can accepts greeting and small talk but always try to steer the conversation towards HR policies and employee benefits.

Rules:
- Only use information from the provided context.
- If the answer is not available in the context, say:
  "I could not find this information in the HR policy documents."
- Do not make up policies, benefits, rules, or numbers.
- Keep answers professional, concise, and easy to understand.
- If possible, mention the relevant policy section or document name.
- If the policy is unclear or incomplete, clearly say so.
- Do not provide legal advice.

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


# ── Stream generator ─────────────────────────────────────────────────────────
def _stream(question: str):
    try:
        docs = _vector_store.similarity_search(question, k=4)
    except Exception as e:
        yield _sse({"type": "error", "content": f"Vector store unavailable: {e}"})
        yield _sse({"type": "done"})
        return

    if not docs:
        yield _sse({"type": "token", "content": "I don't have sufficient information in the HR policies to answer that."})
        yield _sse({"type": "done"})
        return

    context = "\n\n".join(doc.page_content for doc in docs)
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
    return {"status": "ok", "model": "llama3.2"}

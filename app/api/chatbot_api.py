"""FastAPI router for chatbot endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from app.chatbot.chatbot import get_chatbot

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    model: Optional[str] = "llama2"


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Dict[str, str]]
    model: str
    has_context: bool
    chat_history_length: Optional[int] = None


class SearchRequest(BaseModel):
    """Document search request"""
    query: str
    k: Optional[int] = 5


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the HR assistant

    Args:
        request: Chat request with query and model name

    Returns:
        Chat response with answer and sources
    """
    try:
        chatbot = get_chatbot(model_name=request.model)
        result = chatbot.answer(request.query)
        return ChatResponse(**result)
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/chat/history")
async def get_history():
    """Get conversation history"""
    try:
        chatbot = get_chatbot()
        history = chatbot.get_conversation_history()
        return {
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.delete("/chat/history")
async def clear_history():
    """Clear conversation history"""
    try:
        chatbot = get_chatbot()
        chatbot.clear_memory()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing history: {str(e)}"
        )


@router.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search for relevant HR policy documents

    Args:
        request: Search request with query and number of results

    Returns:
        List of relevant documents with metadata
    """
    try:
        chatbot = get_chatbot()
        results = chatbot.search_documents(request.query, k=request.k)
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@router.post("/refresh-db")
async def refresh_database():
    """Refresh vector database after new documents are uploaded"""
    try:
        chatbot = get_chatbot()
        chatbot.update_vector_db()
        return {
            "message": "Vector database refreshed successfully",
            "has_context": chatbot.vector_db is not None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing database: {str(e)}"
        )


@router.get("/health")
async def chatbot_health():
    """Check chatbot health and Ollama connection"""
    try:
        chatbot = get_chatbot()
        return {
            "status": "healthy",
            "model": chatbot.model_name,
            "has_vector_db": chatbot.vector_db is not None,
            "ollama_url": chatbot.ollama_base_url,
            "embeddings_model": chatbot.embeddings_model
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Chatbot health check failed: {str(e)}"
        )

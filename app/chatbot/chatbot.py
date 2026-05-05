"""Chatbot module for HR assistance using Ollama and FAISS"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class HRChatbot:
    """HR Assistance Chatbot using Ollama and FAISS Vector Database"""

    def __init__(
        self,
        model_name: str = "llama3.2",
        vector_db_path: Optional[Path] = None,
        ollama_base_url: str = "http://localhost:11434",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        temperature: float = 0.7,
    ):
        """
        Initialize the chatbot with Ollama LLM and FAISS vector database

        Args:
            model_name: Ollama model to use (e.g., 'llama2', 'mistral', 'neural-chat')
            vector_db_path: Path to FAISS vector database
            ollama_base_url: Base URL for Ollama service
            embeddings_model: HuggingFace embeddings model name
            temperature: LLM temperature for creativity (0.0-1.0)
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.embeddings_model = embeddings_model
        self.temperature = temperature

        # Set default vector DB path if not provided
        if vector_db_path is None:
            base_dir = Path(__file__).parent.parent.parent
            vector_db_path = BASE_DIR / "app" / "data" / "hr_policy_vector_db"

        self.vector_db_path = Path(vector_db_path)

        # Initialize components
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.chain = None
        self.memory = None

        # Initialize all components
        self._initialize()

    def _initialize(self) -> None:
        """Initialize embeddings, vector DB, LLM, and conversation chain"""
        print("Initializing HR Chatbot...")

        # Check Ollama connection
        if not self._check_ollama_connection():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running: ollama serve"
            )

        # Initialize embeddings
        print(f"Loading embeddings model: {self.embeddings_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model
        )

        # Load vector database
        self._load_vector_db()

        # Initialize Ollama LLM
        print(f"Initializing Ollama model: {self.model_name}")
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
        )

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # Create retrieval chain
        if self.vector_db:
            self._create_chain()
        else:
            print("Warning: Vector database not loaded. Chatbot will operate without document context.")

        print("HR Chatbot initialized successfully!")

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _load_vector_db(self) -> None:
        """Load FAISS vector database"""
        if not self.vector_db_path.exists():
            print(f"Warning: Vector database not found at {self.vector_db_path}")
            print("Upload HR policy documents to create the database.")
            self.vector_db = None
            return

        try:
            print(f"Loading vector database from {self.vector_db_path}")
            self.vector_db = FAISS.load_local(
                str(self.vector_db_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Vector database loaded successfully!")
        except Exception as e:
            print(f"Error loading vector database: {e}")
            self.vector_db = None

    def _create_chain(self) -> None:
        """Create conversational retrieval chain"""
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},  # Retrieve top 3 relevant documents
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
        )

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the chatbot

        Args:
            query: User question about HR policies

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.chain:
            if self.llm:
                # Fallback to general LLM if no vector DB
                answer = self.llm.invoke(query)
                return {
                    "answer": answer,
                    "sources": [],
                    "model": self.model_name,
                    "has_context": False,
                }
            else:
                raise RuntimeError("Chatbot not properly initialized")

        try:
            # Get response from chain
            response = self.chain.invoke(
                {"question": query},
                return_only_outputs=False,
            )

            # Extract answer and source documents
            answer = response.get("answer", "I couldn't generate a response.")
            source_docs = response.get("source_documents", [])

            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "category": doc.metadata.get("category", "unknown"),
                })

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_name,
                "has_context": True,
                "chat_history_length": len(self.memory.buffer_as_messages) if hasattr(self.memory, 'buffer_as_messages') else 0,
            }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "model": self.model_name,
                "has_context": False,
                "error": str(e),
            }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        history = []
        if self.memory:
            try:
                messages = self.memory.buffer_as_messages if hasattr(self.memory, 'buffer_as_messages') else []
                for msg in messages:
                    history.append({
                        "role": msg.type if hasattr(msg, 'type') else msg.__class__.__name__,
                        "content": msg.content,
                    })
            except Exception as e:
                print(f"Error retrieving conversation history: {e}")
        return history

    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("Conversation history cleared.")

    def update_vector_db(self) -> None:
        """Reload vector database (useful after new documents are uploaded)"""
        print("Reloading vector database...")
        self._load_vector_db()
        if self.vector_db:
            self._create_chain()
            print("Vector database updated and chain recreated.")

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Search for relevant documents without generating answer"""
        if not self.vector_db:
            return []

        try:
            results = self.vector_db.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "category": doc.metadata.get("category", "unknown"),
                }
                for doc in results
            ]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []


# Singleton instance for application-wide use
_chatbot_instance: Optional[HRChatbot] = None


def get_chatbot(
    model_name: str = "llama2",
    vector_db_path: Optional[Path] = None,
    ollama_base_url: str = "http://localhost:11434",
) -> HRChatbot:
    """
    Get or create the singleton chatbot instance

    Args:
        model_name: Ollama model to use
        vector_db_path: Path to vector database
        ollama_base_url: Ollama service URL

    Returns:
        HRChatbot instance
    """
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = HRChatbot(
            model_name=model_name,
            vector_db_path=vector_db_path,
            ollama_base_url=ollama_base_url,
        )
    return _chatbot_instance


if __name__ == "__main__":
    # Example usage
    try:
        chatbot = HRChatbot(model_name="llama2")

        # Example questions
        questions = [
            "What is the leave policy?",
            "How many days of annual leave do employees get?",
            "What are the safety requirements?",
        ]

        for question in questions:
            print(f"\n{'='*60}")
            print(f"Q: {question}")
            result = chatbot.answer(question)
            print(f"\nA: {result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  {i}. [{source['category']}] {source['content'][:100]}...")

        # Show conversation history
        print(f"\n{'='*60}")
        print("Conversation History:")
        for msg in chatbot.get_conversation_history():
            print(f"  [{msg['role']}]: {msg['content'][:80]}...")

    except ConnectionError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Download Ollama from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. In another terminal, pull a model: ollama pull llama2")
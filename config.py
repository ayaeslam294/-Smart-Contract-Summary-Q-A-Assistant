import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# File Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_llm():
    """Returns a Groq LLM client."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )

def get_embedder():
    """Returns a local HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

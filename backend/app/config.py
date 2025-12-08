"""
Configuration settings for the Hybrid Buyer Advisor application.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    
    # OpenAI Model Settings
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000
    
    # Local Embedding Model (alternative to OpenAI)
    USE_LOCAL_EMBEDDINGS: bool = True
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "superlinked"  # Options: "superlinked", "qdrant", "pinecone"
    USE_SUPERLINKED: bool = True  # Use Superlinked for vector search (recommended)
    
    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = ""  # Full URL (optional, overrides host:port if set)
    QDRANT_API_KEY: str = ""  # API key for Qdrant Cloud (leave empty for local)
    QDRANT_COLLECTION_NAME: str = "realestate"
    QDRANT_PATH: str = "./qdrant_data"  # For local file-based storage
    QDRANT_USE_LOCAL_FILE: bool = False  # Use file storage instead of server
    
    # Pinecone Settings (if using Pinecone)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "realestate-index"
    
    # Embedding Dimensions
    EMBEDDING_DIMENSION: int = 384  # 384 for MiniLM, 1536 for OpenAI ada-002
    
    # Search Settings
    SEARCH_TOP_K: int = 5
    
    # Data Settings
    DATA_PATH: str = "./data"
    DATASET_FILENAME: str = "realtor-data.csv"  # Default dataset
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    SYSTEM_PROMPTS_DIR: Path = BASE_DIR / "System_Prompts"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to load prompt files
def load_system_prompt(agent_name: str) -> str:
    """
    Load a system prompt from the System_Prompts directory.
    
    Args:
        agent_name: Name of the agent (e.g., 'router_agent', 'filter_agent')
        
    Returns:
        The prompt text content
    """
    settings = get_settings()
    prompt_file = settings.SYSTEM_PROMPTS_DIR / f"{agent_name}_prompt.txt"
    
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    else:
        return f"[PLACEHOLDER] System prompt for {agent_name} not found at {prompt_file}"


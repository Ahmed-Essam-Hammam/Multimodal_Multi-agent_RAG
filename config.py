"""
Configuration settings for the multimodal RAG system.
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    CEREBRAS_API_KEY: str

    # Model Configuration
    MODEL_NAME: str = "llama-3.3-70b"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048

    # Vector Store Configuration
    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "multimodal_rag"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Multimodal Embeddings (CLIP)
    USE_CLIP_EMBEDDINGS: bool = True  # Use CLIP for true multimodal embeddings
    CLIP_MODEL: str = "openai/clip-vit-large-patch14"  # CLIP model variant
    # Options: 
    # - openai/clip-vit-base-patch32 (default, balanced)
    # - openai/clip-vit-large-patch14 (better quality, slower)
    # - openai/clip-vit-base-patch16 (faster)

    HYBRID_EMBEDDINGS: bool = True  # Use CLIP for images, OpenAI for text
    CLIP_FOR_TEXT_SEARCH: bool = True  # Use CLIP text encoder for cross-modal search
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Document Processing
    UPLOAD_DIR: str = "./data/uploads"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Image Processing
    IMAGE_EXTRACT_DPI: int = 300
    MAX_IMAGE_SIZE: tuple = (1024, 1024)

    # Agent Configuration
    MAX_ITERATIONS: int = 5
    ENABLE_VERIFICATION: bool = True


    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
"""
Tools for retrieving information from the vector store.
"""

from typing import List, Dict, Any
from langchain.tools import tool
from utils.vector_store import VectorStoreManager


class RetrievalTools:
    """Tools for retrieving text and images from the vector store."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve_multimodal(self, query: str, text_top_k: int = 5, image_top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve both text and images relevant to the query.
        """
        texts = self.vector_store.search_text(query, text_top_k)
        images = self.vector_store.search_images(query, image_top_k)
        
        return {
            "texts": texts,
            "images": images
        }        
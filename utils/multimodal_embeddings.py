"""
Enhanced multimodal embeddings using CLIP for proper image-text matching.

This module provides true multimodal embeddings that can match:
- Text queries with text chunks
- Text queries with images
- Image queries with images
- Image queries with text
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import base64
from io import BytesIO
import torch.nn.functional as F


try:
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

from config import settings

class MultimodalEmbeddings:
    """
    Multimodal embedding generator using CLIP.
    
    CLIP (Contrastive Language-Image Pre-training) creates embeddings in a shared
    space where semantically similar images and text are close together.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize CLIP model for multimodal embeddings.
        
        Args:
            model_name: HuggingFace model name for CLIP
        """

        if not CLIP_AVAILABLE:
            raise ImportError("transformers package required. Install with: pip install transformers torch")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Set to evaluation mode
        self.model.eval()
        
        print(f"✓ CLIP model loaded: {model_name}")


    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using CLIP text encoder.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings (normalized)
        """

        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            # Tokenize and encode
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)


            
            outputs = self.model.text_model(**inputs)            
            text_features = outputs.pooler_output                
            text_features = self.model.text_projection(text_features)     
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu().numpy()
    

    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for images using CLIP image encoder.
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            Numpy array of embeddings (normalized)
        """
        if isinstance(images, Image.Image):
            images = [images]


        with torch.no_grad():
            # Process images
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            # Get image embeddings
            outputs = self.model.vision_model(**inputs)           
            image_features = outputs.pooler_output                
            image_features = self.model.visual_projection(image_features)  
            image_features = F.normalize(image_features, p=2, dim=-1) 
            
        return image_features.cpu().numpy()
    

    def embed_image_from_path(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of embedding (normalized)
        """

        image = Image.open(image_path).convert("RGB")
        return self.embed_images(image)[0]


    def embed_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate embedding for image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Numpy array of embedding (normalized)
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.embed_images(image)[0]   
        

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and candidates.
        
        Args:
            query_embedding: Query embedding (1D array)
            candidate_embeddings: Candidate embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """       

        # Ensure 2D arrays
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if candidate_embeddings.ndim == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)

        # Compute cosine similarity
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

        return similarities
    

    # def rank_by_similarity(
    #     self,
    #     query_embedding: np.ndarray,
    #     candidate_embeddings: List[np.ndarray],
    #     metadata: List[Dict[str, Any]],
    #     top_k: int = 5
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Rank candidates by similarity to query.
        
    #     Args:
    #         query_embedding: Query embedding
    #         candidate_embeddings: List of candidate embeddings
    #         metadata: Metadata for each candidate
    #         top_k: Number of top results to return
            
    #     Returns:
    #         List of top-k results with metadata and scores
    #     """
    #     if not candidate_embeddings:
    #         return []
        
    #     # Stack embeddings
    #     embeddings_array = np.vstack(candidate_embeddings)

    #     # Compute similarities
    #     similarities = self.compute_similarity(query_embedding, embeddings_array)
        
    #     # Get top-k indices
    #     top_indices = np.argsort(similarities)[::-1][:top_k]

    #     # Prepare results
    #     results = []
    #     for idx in top_indices:
    #         result = metadata[idx].copy()
    #         result['similarity'] = float(similarities[idx])
    #         result['embedding'] = candidate_embeddings[idx]
    #         results.append(result)
        
    #     return results
    


class HybridEmbeddings:
    """
    Hybrid embeddings that use CLIP for multimodal and OpenAI for text-only.
    
    This provides best of both worlds:
    - CLIP for image-text matching
    - OpenAI embeddings for pure text search (often better for text-only)
    """
    
    def __init__(
        self,
        use_clip: bool = True,
        clip_model: str = "openai/clip-vit-large-patch14"
    ):
        """
        Initialize hybrid embeddings.
        
        Args:
            use_clip: Whether to use CLIP for multimodal
            clip_model: CLIP model name
        """
        self.use_clip = use_clip and CLIP_AVAILABLE
        
        if self.use_clip:
            self.clip = MultimodalEmbeddings(clip_model)
        else:
            self.clip = None
        
        # OpenAI embeddings for text (fallback)
        try:
            from langchain_openai import OpenAIEmbeddings
            self.text_embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.CEREBRAS_API_KEY,
                openai_api_base="https://api.cerebras.ai/v1"
            )
        except:
            self.text_embeddings = None

    def embed_text_for_retrieval(self, text: str, use_clip: bool = False) -> np.ndarray:
        """
        Embed text for retrieval.
        
        Args:
            text: Text to embed
            use_clip: Force use of CLIP even for text-only
            
        Returns:
            Embedding vector
        """

        if use_clip and self.use_clip:
            return self.clip.embed_text(text)[0]
        elif self.text_embeddings:
            return np.array(self.text_embeddings.embed_query(text))
        elif self.use_clip:
            return self.clip.embed_text(text)[0]
        else:
            raise ValueError("No embedding model available")
        

    def embed_image_for_retrieval(self, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Embed image for retrieval.
        
        Args:
            image: Image as path, bytes, or PIL Image
            
        Returns:
            Embedding vector
        """
        if not self.use_clip:
            raise ValueError("CLIP not available for image embeddings")
        
        if isinstance(image, str):
            return self.clip.embed_image_from_path(image)
        elif isinstance(image, bytes):
            return self.clip.embed_image_from_bytes(image)
        elif isinstance(image, Image.Image):
            return self.clip.embed_images(image)[0]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        

    # def search_multimodal(
    #     self,
    #     query: Union[str, Image.Image, bytes],
    #     text_chunks: List[Dict[str, Any]],
    #     images: List[Dict[str, Any]],
    #     top_k: int = 5,
    #     text_weight: float = 0.5
    # ) -> Dict[str, List[Dict[str, Any]]]:
    #     """
    #     Search across both text and images using multimodal embeddings.
        
    #     Args:
    #         query: Text query or image query
    #         text_chunks: List of text chunks with embeddings
    #         images: List of images with embeddings
    #         top_k: Number of results per modality
    #         text_weight: Weight for text vs images (0-1)
            
    #     Returns:
    #         Dictionary with 'texts' and 'images' results
    #     """
    #     if not self.use_clip:
    #         raise ValueError("CLIP required for multimodal search")
        
    #     # Embed query
    #     if isinstance(query, str):
    #         query_embedding = self.clip.embed_text(query)[0]
    #     else:
    #         query_embedding = self.embed_image_for_retrieval(query)
        
    #     # Search texts
    #     text_results = []
    #     if text_chunks:
    #         text_embeddings = [chunk['embedding'] for chunk in text_chunks]
    #         text_metadata = [
    #             {
    #                 'content': chunk.get('content', ''),
    #                 'metadata': chunk.get('metadata', {}),
    #                 'id': chunk.get('id', '')
    #             }
    #             for chunk in text_chunks
    #         ]
    #         text_results = self.clip.rank_by_similarity(
    #             query_embedding,
    #             text_embeddings,
    #             text_metadata,
    #             top_k
    #         )
        
    #     # Search images
    #     image_results = []
    #     if images:
    #         image_embeddings = [img['embedding'] for img in images]
    #         image_metadata = [
    #             {
    #                 'path': img.get('path', ''),
    #                 'metadata': img.get('metadata', {}),
    #                 'id': img.get('id', '')
    #             }
    #             for img in images
    #         ]
    #         image_results = self.clip.rank_by_similarity(
    #             query_embedding,
    #             image_embeddings,
    #             image_metadata,
    #             top_k
    #         )
        
    #     return {
    #         'texts': text_results,
    #         'images': image_results
    #     }


def compute_contrastive_similarity(
    text_embedding: np.ndarray,
    image_embedding: np.ndarray
) -> float:
    """
    Compute contrastive similarity between text and image embeddings.
    
    This is the core of CLIP's power - embeddings are in the same space.
    
    Args:
        text_embedding: Text embedding from CLIP
        image_embedding: Image embedding from CLIP
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize if not already
    text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
    image_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    
    # Cosine similarity
    similarity = np.dot(text_norm, image_norm)
    
    # Scale to 0-1
    similarity = (similarity + 1) / 2
    
    return float(similarity)
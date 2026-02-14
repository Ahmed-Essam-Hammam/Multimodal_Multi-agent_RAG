"""
Image search integration using existing CLIP multimodal embeddings.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from utils.multimodal_embeddings import MultimodalEmbeddings
from config import settings


class ImageSearch:
    """Image search using CLIP from multimodal_embeddings."""
    
    def __init__(self):
        """Initialize with your existing CLIP model."""
        try:
            # Use smaller, faster model
            self.clip = MultimodalEmbeddings(model_name="openai/clip-vit-large-patch14")
            self.available = True
        except Exception as e:
            print(f"⚠️ CLIP not available: {e}")
            self.available = False
    
    def search_similar_images(
        self,
        query_image_path: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar images in the database.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            List of similar images with similarity scores
        """
        if not self.available:
            print("❌ CLIP not available!")
            return []
        
        # Get all extracted images
        image_dir = os.path.join(settings.UPLOAD_DIR, "extracted_images")
        if not os.path.exists(image_dir):
            print("❌ No extracted images found!")
            return []
        
        # Find all images
        image_files = list(Path(image_dir).glob("*.png")) + \
                     list(Path(image_dir).glob("*.jpg"))
        
        if not image_files:
            print("❌ No images in database!")
            return []
        
        print(f"\n🔍 Searching {len(image_files)} images using CLIP...")
        
        # Embed query image
        print(f"Embedding query: {query_image_path}")
        query_embedding = self.clip.embed_image_from_path(query_image_path)
        
        # Embed and compare all database images
        results = []
        for img_path in image_files:
            try:
                # Embed database image
                db_embedding = self.clip.embed_image_from_path(str(img_path))
                
                # Compute similarity
                similarity = self.clip.compute_similarity(query_embedding, db_embedding)[0]
                
                results.append({
                    "path": str(img_path),
                    "similarity": float(similarity),
                    "filename": img_path.name
                })
                
            except Exception as e:
                print(f"  ⚠️ Error with {img_path.name}: {e}")
                continue
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k
        return results[:top_k]
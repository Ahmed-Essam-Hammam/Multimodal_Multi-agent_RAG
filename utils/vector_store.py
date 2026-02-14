"""
Enhanced vector store with fallback embedding options.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from PIL import Image
from config import settings

# Import CLIP
try:
    from utils.multimodal_embeddings import MultimodalEmbeddings
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available - falling back to text embeddings")

# Fallback to text embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except:
    HUGGINGFACE_AVAILABLE = False


class VectorStoreManager:
    """Vector store using CLIP for multimodal embeddings."""
    
    def __init__(self, use_clip: bool = True):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize embeddings
        self.use_clip = use_clip and CLIP_AVAILABLE
        self.embeddings = self._initialize_embeddings()
        
        self.text_collection = self._get_or_create_collection("text_chunks")
        self.image_collection = self._get_or_create_collection("image_metadata")
    
    def _initialize_embeddings(self):
        """Initialize CLIP or fallback embeddings."""
        
        if self.use_clip:
            try:
                clip = MultimodalEmbeddings(model_name="openai/clip-vit-large-patch14")
                print("✓ Using CLIP for multimodal embeddings")
                return clip
            except Exception as e:
                print(f"⚠️ CLIP failed: {e}")
                self.use_clip = False
        
        # Fallback to text-only embeddings
        if HUGGINGFACE_AVAILABLE:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✓ Using HuggingFace text embeddings (fallback)")
            return embeddings
        
        raise RuntimeError("No embedding model available!")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a Chroma collection."""
        try:
            collection = self.chroma_client.get_collection(name=name)
        except:
            collection = self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return collection
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add text documents to the vector store."""
        if not documents:
            return []
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        print(f"Generating embeddings for {len(texts)} documents...")
        
        # Use CLIP or fallback
        if self.use_clip:
            # CLIP text encoder
            embeddings = [self.embeddings.embed_text(text)[0].tolist() for text in texts]
        else:
            # Fallback text embeddings
            embeddings = self.embeddings.embed_documents(texts)
        
        self.text_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def add_images(self, images: List[Dict[str, Any]]) -> List[str]:
        """
        Add images to vector store.
        
        With CLIP: Embeds actual image pixels (visual features)
        Without CLIP: Embeds text description
        """
        if not images:
            return []
        
        ids = []
        embeddings_list = []
        documents = []
        metadatas = []
        
        print(f"Generating embeddings for {len(images)} images...")
        
        for img in images:
            img_id = hashlib.md5(img["path"].encode()).hexdigest()
            ids.append(img_id)
            
            # Generate description (used as document text)
            description = img.get("description", f"Image from {img.get('source_doc', 'unknown')} page {img.get('page', 'N/A')}")
            documents.append(description)
            
            # FIXED SECTION: PROPERLY USE CLIP VISION ENCODER
            if self.use_clip:
                # ✅ CLIP: Embed the ACTUAL IMAGE PIXELS using vision encoder!
                img_path = img["path"]
                
                if os.path.exists(img_path):
                    try:
                        # CRITICAL FIX: Use embed_image_from_path() for visual embedding!
                        embedding = self.embeddings.embed_image_from_path(img_path).tolist()
                        print(f"  ✅ CLIP vision-embedded: {Path(img_path).name}")
                        
                    except Exception as e:
                        # Only fall back to text on error
                        print(f"  ❌ CLIP vision embedding failed for {Path(img_path).name}: {e}")
                        print(f"     Falling back to text description")
                        embedding = self.embeddings.embed_text(description)[0].tolist()
                else:
                    # Image file doesn't exist
                    print(f"  ⚠️ Image file not found: {img_path}")
                    print(f"     Using text description embedding")
                    embedding = self.embeddings.embed_text(description)[0].tolist()
            else:
                # ❌ Fallback: Embed text description only
                embedding = self.embeddings.embed_query(description)
            
            embeddings_list.append(embedding)
            
            metadatas.append({
                "path": img["path"],
                "source_doc": img.get("source_doc", ""),
                "page": img.get("page", 0),
                "type": "image",
                "description": description
            })
        
        self.image_collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(ids)} images to vector store")
        
        return ids
    
    def search_text(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant text chunks.
        
        With CLIP: Uses CLIP text encoder
        Without CLIP: Uses sentence transformers
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate query embedding
        if self.use_clip:
            # CLIP text encoder
            query_embedding = self.embeddings.embed_text(query)[0].tolist()
        else:
            # Fallback
            query_embedding = self.embeddings.embed_query(query)
        
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": similarity
                    })
        
        return formatted_results
    
    def search_images(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant images based on TEXT query.
        
        THIS IS THE KEY IMPROVEMENT:
        With CLIP: Text query can find visually relevant images!
        Without CLIP: Text query matches text descriptions only.
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate query embedding
        if self.use_clip:
            # 🎯 CLIP: Text query embedded in SAME SPACE as image embeddings!
            query_embedding = self.embeddings.embed_text(query)[0].tolist()
            print(f"  🎨 CLIP text query searching visual image embeddings")
        else:
            # Fallback: Text query searches text descriptions
            query_embedding = self.embeddings.embed_query(query)
        
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "path": results['metadatas'][0][i].get("path", ""),
                        "metadata": results['metadatas'][0][i],
                        "similarity": similarity
                    })
        
        return formatted_results
    
    def search_by_image(self, image_path: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search using an image query.
        
        With CLIP: Image query finds visually similar images!
        Without CLIP: Not supported.
        """
        if not self.use_clip:
            raise ValueError("Image search requires CLIP!")
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # 🎯 CLIP: Embed the query image
        query_embedding = self.embeddings.embed_image_from_path(image_path).tolist()
        print(f"  🎨 CLIP image query searching visual image embeddings")
        
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "path": results['metadatas'][0][i].get("path", ""),
                    "metadata": results['metadatas'][0][i],
                    "similarity": similarity,
                    "filename": Path(results['metadatas'][0][i].get("path", "")).name
                })
        
        # Sort by similarity
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return formatted_results[:top_k]
    
    def reset_store(self):
        """Reset the entire vector store."""
        try:
            self.chroma_client.delete_collection("text_chunks")
            self.chroma_client.delete_collection("image_metadata")
        except:
            pass
        self.text_collection = self._get_or_create_collection("text_chunks")
        self.image_collection = self._get_or_create_collection("image_metadata")
        print("✓ Vector store reset")
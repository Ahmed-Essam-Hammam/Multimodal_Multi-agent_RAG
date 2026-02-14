"""
Main with image search using your existing CLIP module.
Auto-displays images and shows existing files on startup.
"""
import os
import argparse
from pathlib import Path
from typing import List
from graph import MultimodalRAGGraph
from utils import DocumentProcessor
from config import settings
from PIL import Image
import matplotlib.pyplot as plt

# Import image search (using your CLIP)
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.image_search import ImageSearch
    IMAGE_SEARCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Image search not available: {e}")
    IMAGE_SEARCH_AVAILABLE = False


class MultimodalRAGSystem:
    """Main system with CLIP image search."""
    
    def __init__(self):
        self.graph = MultimodalRAGGraph()
        self.doc_processor = DocumentProcessor()
        self.vector_store = self.graph.get_vector_store()
        
        # Initialize image search
        if IMAGE_SEARCH_AVAILABLE:
            print("\n🎨 Initializing CLIP image search...")
            self.image_search = ImageSearch()
        else:
            self.image_search = None
            print("⚠️ Image search not available")
        
        # Show existing files
        self._show_existing_files()
    
    def _show_existing_files(self):
        """Display existing files in the system."""
        print("\n" + "="*80)
        print("📁 EXISTING FILES IN DATABASE")
        print("="*80)
        
        # Check for extracted images
        image_dir = os.path.join(settings.UPLOAD_DIR, "extracted_images")
        if os.path.exists(image_dir):
            image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
            if image_files:
                print(f"\n✓ Found {len(image_files)} extracted images:")
                
                # Group by source document
                sources = {}
                for img_path in image_files:
                    # Extract source from filename (e.g., "Attention_page3_img1.png" -> "Attention")
                    name = img_path.stem
                    source = name.split('_page')[0] if '_page' in name else name.split('_img')[0]
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(img_path.name)
                
                for source, imgs in sources.items():
                    print(f"\n  📄 {source}:")
                    for img in sorted(imgs)[:5]:  # Show first 5 images per source
                        print(f"      - {img}")
                    if len(imgs) > 5:
                        print(f"      ... and {len(imgs) - 5} more")
            else:
                print("\n⚠️ No extracted images found")
        else:
            print("\n⚠️ No extracted images directory found")
        
        # Check ChromaDB collections
        try:
            text_count = self.vector_store.text_collection.count()
            image_count = self.vector_store.image_collection.count()
            
            print(f"\n📊 Vector Database:")
            print(f"   Text chunks: {text_count}")
            print(f"   Image entries: {image_count}")
            
            if text_count == 0 and image_count == 0:
                print("\n💡 Tip: Use --mode ingest --files <your_files> to add documents")
        except Exception as e:
            print(f"\n⚠️ Could not read database: {e}")
        
        print("="*80 + "\n")
    
    def ingest_documents(self, file_paths: List[str]):
        """Ingest documents."""
        print("\n=== DOCUMENT INGESTION ===")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                continue
            
            print(f"\nProcessing: {file_path}")
            
            try:
                documents, images = self.doc_processor.process_document(file_path)
                text_ids = self.vector_store.add_documents(documents)
                image_ids = self.vector_store.add_images(images)
                
                print(f"  ✓ Added {len(text_ids)} text chunks")
                print(f"  ✓ Added {len(image_ids)} images")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print("\n=== INGESTION COMPLETE ===\n")
    
    def search_by_image(self, query_image_path: str, top_k: int = 5):
        """Search using an image query - ALWAYS displays results."""
        if not self.image_search or not self.image_search.available:
            print("\n❌ Image search not available!")
            print("Install: pip install transformers torch")
            return []
        
        print("\n" + "="*80)
        print("IMAGE SIMILARITY SEARCH")
        print("="*80)
        
        # Perform search
        results = self.image_search.search_similar_images(query_image_path, top_k)
        
        if results:
            print(f"\n✓ Found {len(results)} similar images:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity']:.1%}")
                print(f"   File: {result['filename']}")
                print(f"   Path: {result['path']}\n")
            
            # ALWAYS display (no asking)
            self._display_image_results(query_image_path, results)
            
            return results
        else:
            print("\n❌ No similar images found!")
            return []
    
    def _display_image_results(self, query_path: str, results: list):
        """Display image search results in matplotlib."""
        try:
            num_results = len(results)
            num_total = num_results + 1  # +1 for query
            
            # Layout
            cols = min(3, num_total)
            rows = (num_total + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            
            # Handle single subplot
            if num_total == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Show query image
            query_img = Image.open(query_path)
            axes[0].imshow(query_img)
            axes[0].set_title("🔍 Query Image", fontsize=16, fontweight='bold', color='blue')
            axes[0].axis('off')
            axes[0].set_facecolor('#e8f4f8')
            
            # Show results
            for idx, result in enumerate(results, 1):
                img = Image.open(result['path'])
                axes[idx].imshow(img)
                
                # Color-code by similarity
                if result['similarity'] > 0.8:
                    color = 'green'
                    badge = '✅'
                elif result['similarity'] > 0.6:
                    color = 'orange'
                    badge = '⚠️'
                else:
                    color = 'red'
                    badge = '❌'
                
                axes[idx].set_title(
                    f"{badge} Match {idx}\nSimilarity: {result['similarity']:.1%}",
                    fontsize=14,
                    color=color,
                    fontweight='bold'
                )
                axes[idx].axis('off')
            
            # Hide unused subplots
            for idx in range(num_total, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(
                "CLIP Image Similarity Search Results",
                fontsize=18,
                fontweight='bold',
                y=0.98
            )
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def query(self, query_text: str, query_type: str = "text", image_path: str = None):
        """Query the RAG system."""
        
        # Image search mode
        if image_path and os.path.exists(image_path):
            return self.search_by_image(image_path)
        
        # Regular text query
        print("\n" + "="*80)
        print(f"QUERY: {query_text}")
        print("="*80)
        
        result = self.graph.run(
            query=query_text,
            query_type=query_type,
            query_image=None
        )
        
        self._display_results(result)
        return result
    
    def _display_results(self, result: dict):
        """Display query results - ALWAYS shows images."""
        print("\n" + "-"*80)
        print("RETRIEVAL")
        print("-"*80)
        print(f"Texts: {len(result['retrieved_texts'])}")
        print(f"Images: {len(result['retrieved_images'])}")
        print(f"Success: {result['retrieval_successful']}")
        
        if result['retrieval_successful']:
            if result['retrieved_texts']:
                print("\nTop texts:")
                for i, text in enumerate(result['retrieved_texts'][:3], 1):
                    print(f"  {i}. Similarity: {text['similarity']:.2%}")
                    print(f"     {text['content'][:100]}...")
            
            if result['retrieved_images']:
                print("\nImages:")
                for i, img in enumerate(result['retrieved_images'], 1):
                    print(f"  {i}. {img['path']} ({img['similarity']:.2%})")
                
                # ALWAYS display (no asking)
                self._show_images(result['retrieved_images'])
        
        print("\n" + "-"*80)
        print("ANSWER")
        print("-"*80)
        print(result['final_answer'])
        print("\n" + "="*80 + "\n")
    
    def _show_images(self, images: list):
        """Display retrieved images."""
        try:
            num = len(images)
            fig, axes = plt.subplots(1, num, figsize=(5*num, 5))
            if num == 1:
                axes = [axes]
            
            for idx, img_data in enumerate(images):
                if os.path.exists(img_data['path']):
                    img = Image.open(img_data['path'])
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"Image {idx+1}\n{img_data['similarity']:.1%}")
                    axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
    
    def reset(self):
        """Reset vector store."""
        print("\nResetting...")
        self.vector_store.reset_store()
        print("✓ Complete\n")


def main():
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, 
                       choices=["ingest", "query", "image_search", "interactive", "reset"])
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--query", type=str)
    parser.add_argument("--image", type=str)
    
    args = parser.parse_args()
    system = MultimodalRAGSystem()
    
    if args.mode == "ingest":
        if not args.files:
            print("Error: --files required")
            return
        system.ingest_documents(args.files)
    
    elif args.mode == "query":
        if not args.query:
            print("Error: --query required")
            return
        system.query(args.query, image_path=args.image)
    
    elif args.mode == "image_search":
        if not args.image:
            print("Error: --image required")
            return
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        system.search_by_image(args.image)
    
    elif args.mode == "reset":
        confirm = input("Reset? (yes/no): ")
        if confirm.lower() == "yes":
            system.reset()
    
    elif args.mode == "interactive":
        print("\n" + "="*80)
        print("MULTIMODAL RAG - INTERACTIVE")
        print("="*80)
        print("\nCommands:")
        print("  <text>              - Text search")
        print("  image <path>        - Image search")
        print("  ingest <file>       - Add document")
        print("  reset               - Clear database")
        print("  exit                - Quit")
        print("="*80 + "\n")
        
        while True:
            try:
                inp = input("\n> ").strip()
                if not inp:
                    continue
                
                if inp.lower() == "exit":
                    break
                elif inp.lower() == "reset":
                    confirm = input("Confirm? (yes/no): ")
                    if confirm.lower() == "yes":
                        system.reset()
                elif inp.startswith("ingest "):
                    system.ingest_documents([inp[7:].strip()])
                elif inp.startswith("image "):
                    img_path = inp[6:].strip()
                    if os.path.exists(img_path):
                        system.search_by_image(img_path)
                    else:
                        print(f"Image not found: {img_path}")
                else:
                    system.query(inp)
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
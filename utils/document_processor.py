"""
Document processor that filters out duplicate and inverted images.
"""
import os
import io
import base64
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from config import settings


class DocumentProcessor:
    """Process documents and filter out duplicate/inverted images."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def process_document(self, file_path: str) -> Tuple[List[LangChainDocument], List[Dict[str, Any]]]:
        """Process a document and extract text chunks and images."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self._process_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _is_image_too_dark(self, img: Image.Image) -> bool:
        """Check if image is too dark (inverted/negative)."""
        img_array = np.array(img.convert('RGB'))
        mean_brightness = np.mean(img_array)
        return mean_brightness < 20  # Threshold for dark images
    
    def _is_image_too_light(self, img: Image.Image) -> bool:
        """Check if image is almost entirely white."""
        img_array = np.array(img.convert('RGB'))
        mean_brightness = np.mean(img_array)
        return mean_brightness > 350  # Almost pure white
    
    def _images_are_similar(self, img1: Image.Image, img2: Image.Image) -> bool:
        """Check if two images are similar (duplicates)."""
        try:
            # Resize both to same size for comparison
            size = (100, 100)
            img1_small = img1.resize(size)
            img2_small = img2.resize(size)
            
            # Convert to same mode
            img1_small = img1_small.convert('RGB')
            img2_small = img2_small.convert('RGB')
            
            # Calculate difference
            diff = ImageChops.difference(img1_small, img2_small)
            diff_array = np.array(diff)
            mean_diff = np.mean(diff_array)
            
            # If difference is very small, they're duplicates
            return mean_diff < 10
        except:
            return False
    
    def _process_pdf(self, file_path: str) -> Tuple[List[LangChainDocument], List[Dict[str, Any]]]:
        """Extract text and images from PDF with filtering."""
        text_chunks = []
        images = []
        
        doc = fitz.open(file_path)
        full_text = ""
        
        doc_name = Path(file_path).stem.replace('_', ' ').replace('-', ' ')
        
        # Store processed images to detect duplicates
        processed_images = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            page_context = page_text[:500] if page_text else ""
            
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB
                    if image.mode not in ('RGB', 'RGBA'):
                        image = image.convert('RGB')
                    
                    # FILTER 1: Skip too dark images (inverted/negative)
                    if self._is_image_too_dark(image):
                        print(f"  ⚠️ Skipping dark/inverted image on page {page_num + 1}")
                        continue
                    
                    # FILTER 2: Skip almost entirely white images
                    if self._is_image_too_light(image):
                        print(f"  ⚠️ Skipping nearly blank image on page {page_num + 1}")
                        continue
                    
                    # FILTER 3: Skip duplicates
                    is_duplicate = False
                    for prev_img in processed_images:
                        if self._images_are_similar(image, prev_img):
                            print(f"  ⚠️ Skipping duplicate image on page {page_num + 1}")
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                    
                    # FILTER 4: Skip very small images (likely icons/decorations)
                    if image.size[0] < 100 or image.size[1] < 100:
                        print(f"  ⚠️ Skipping small image ({image.size[0]}x{image.size[1]}) on page {page_num + 1}")
                        continue
                    
                    # Image passed all filters!
                    processed_images.append(image.copy())
                    
                    # Resize if needed
                    max_size = settings.MAX_IMAGE_SIZE
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Save
                    image_filename = f"{Path(file_path).stem}_page{page_num + 1}_img{len(images) + 1}.png"
                    image_path = os.path.join(settings.UPLOAD_DIR, "extracted_images", image_filename)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image.save(image_path, 'PNG', optimize=False)
                    
                    # Generate description
                    description = self._generate_image_description(
                        doc_name=doc_name,
                        page_num=page_num + 1,
                        img_index=len(images) + 1,
                        page_context=page_context,
                        image_size=image.size
                    )
                    
                    images.append({
                        "path": image_path,
                        "page": page_num + 1,
                        "index": len(images),
                        "source_doc": file_path,
                        "image_bytes": image_bytes,
                        "description": description,
                        "width": image.size[0],
                        "height": image.size[1]
                    })
                    
                    print(f"  ✓ Extracted image {len(images)} from page {page_num + 1}")
                    
                except Exception as e:
                    print(f"  ⚠️ Error extracting image from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        
        documents = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": file_path, "type": "pdf"}]
        )
        
        return documents, images
    
    def _generate_image_description(
        self, 
        doc_name: str, 
        page_num: int, 
        img_index: int,
        page_context: str,
        image_size: tuple
    ) -> str:
        """Generate rich, searchable image description."""
        doc_keywords = doc_name.lower()
        context_lower = page_context.lower()
        
        diagram_terms = []
        if any(term in context_lower for term in ['figure', 'diagram', 'chart', 'graph', 'illustration']):
            diagram_terms.append("diagram")
        if any(term in context_lower for term in ['architecture', 'model', 'structure']):
            diagram_terms.append("architecture")
        if any(term in context_lower for term in ['flow', 'process', 'pipeline']):
            diagram_terms.append("flowchart")
        if any(term in context_lower for term in ['table', 'comparison']):
            diagram_terms.append("table")
        
        is_large = image_size[0] > 400 and image_size[1] > 300
        
        if is_large:
            diagram_terms.append("figure")
        
        parts = []
        if diagram_terms:
            parts.append(" ".join(set(diagram_terms)))
        else:
            parts.append("image figure visualization")
        
        parts.append(f"from {doc_name}")
        parts.append(f"page {page_num}")
        
        context_words = [w for w in page_context.split()[:30] if len(w) > 4]
        if context_words:
            parts.append(" ".join(context_words[:10]))
        
        if is_large:
            parts.append("large detailed figure")
        
        description = " ".join(parts)
        
        if "diagram" not in description.lower() and is_large:
            description = "diagram " + description
        
        return description
    
    def _process_docx(self, file_path: str) -> Tuple[List[LangChainDocument], List[Dict[str, Any]]]:
        """Extract text and images from DOCX."""
        doc = Document(file_path)
        full_text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        images = []
        
        doc_name = Path(file_path).stem.replace('_', ' ').replace('-', ' ')
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image = Image.open(io.BytesIO(image_data))
                    
                    if image.mode not in ('RGB', 'RGBA'):
                        image = image.convert('RGB')
                    
                    # Apply same filters
                    if self._is_image_too_dark(image):
                        continue
                    if self._is_image_too_light(image):
                        continue
                    if image.size[0] < 100 or image.size[1] < 100:
                        continue
                    
                    max_size = settings.MAX_IMAGE_SIZE
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    image_filename = f"{Path(file_path).stem}_img{len(images) + 1}.png"
                    image_path = os.path.join(settings.UPLOAD_DIR, "extracted_images", image_filename)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image.save(image_path, 'PNG', optimize=False)
                    
                    description = f"diagram figure image from {doc_name} visualization chart illustration architecture"
                    
                    images.append({
                        "path": image_path,
                        "index": len(images),
                        "source_doc": file_path,
                        "image_bytes": image_data,
                        "description": description,
                        "width": image.size[0],
                        "height": image.size[1]
                    })
                except Exception as e:
                    continue
        
        documents = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": file_path, "type": "docx"}]
        )
        
        return documents, images
    
    def _process_text(self, file_path: str) -> Tuple[List[LangChainDocument], List[Dict[str, Any]]]:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        documents = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": file_path, "type": "text"}]
        )
        
        return documents, []
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
"""
Streamlit Web Interface with CLIP Image Search Support
Shows existing files on startup and in sidebar
"""
import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageOps

from main import MultimodalRAGSystem
from config import settings

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .source-card {
        background-color: #1e3a5f;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: #ffffff;
    }
    .verification-verified {
        background-color: #1e4620;
        color: #90ee90;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .verification-unverified {
        background-color: #4a3c1a;
        color: #ffd700;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .verification-hallucination {
        background-color: #4a1f1f;
        color: #ffcccb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .chat-message-user {
        background-color: #1e3a5f;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .chat-message-assistant {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, 
    .stMarkdown h3, .stMarkdown h4, .stMarkdown strong {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'image_search_results' not in st.session_state:
    st.session_state.image_search_results = None
if 'existing_files_loaded' not in st.session_state:
    st.session_state.existing_files_loaded = False


def get_existing_files():
    """Get existing files from the database."""
    existing_files = {
        'sources': {},
        'total_images': 0,
        'text_count': 0,
        'image_count': 0
    }
    
    try:
        # Check for extracted images
        image_dir = os.path.join(settings.UPLOAD_DIR, "extracted_images")
        if os.path.exists(image_dir):
            image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
            existing_files['total_images'] = len(image_files)
            
            # Group by source document
            for img_path in image_files:
                name = img_path.stem
                source = name.split('_page')[0] if '_page' in name else name.split('_img')[0]
                if source not in existing_files['sources']:
                    existing_files['sources'][source] = []
                existing_files['sources'][source].append(img_path.name)
        
        # Get database counts if system is initialized
        if st.session_state.rag_system:
            try:
                existing_files['text_count'] = st.session_state.rag_system.vector_store.text_collection.count()
                existing_files['image_count'] = st.session_state.rag_system.vector_store.image_collection.count()
            except:
                pass
    
    except Exception as e:
        st.error(f"Error loading existing files: {e}")
    
    return existing_files


def initialize_system():
    """Initialize the RAG system."""
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = MultimodalRAGSystem()
                
                # Load existing files info
                existing = get_existing_files()
                if existing['sources']:
                    st.session_state.ingested_files = list(existing['sources'].keys())
                    st.session_state.existing_files_loaded = True
                
                st.success("✅ System initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")


def format_verification_status(result: Dict[str, Any]) -> str:
    """Format verification status with appropriate styling."""
    status = result.get('verification_status', 'unknown')
    confidence = result.get('confidence_score', 0.0)
    
    if status == 'verified':
        return f"""
        <div class='verification-verified'>
            <h4 style='color: #90ee90;'>✅ Verified</h4>
            <p style='color: #ffffff;'><strong>Confidence:</strong> {confidence:.1%}</p>
            <div style='background-color: #28a745; height: 20px; width: {confidence*100}%; border-radius: 10px;'></div>
        </div>
        """
    elif status == 'unverified':
        return f"""
        <div class='verification-unverified'>
            <h4 style='color: #ffd700;'>⚠️ Unverified</h4>
            <p style='color: #ffffff;'><strong>Confidence:</strong> {confidence:.1%}</p>
            <div style='background-color: #ffc107; height: 20px; width: {confidence*100}%; border-radius: 10px;'></div>
        </div>
        """
    elif status == 'hallucination_detected':
        return f"""
        <div class='verification-hallucination'>
            <h4 style='color: #ffcccb;'>❌ Hallucination Detected</h4>
            <p style='color: #ffffff;'><strong>Confidence:</strong> {confidence:.1%}</p>
            <div style='background-color: #dc3545; height: 20px; width: {confidence*100}%; border-radius: 10px;'></div>
        </div>
        """
    else:
        return f"""
        <div class='metric-card'>
            <p style='color: #ffffff;'>Status: {status}</p>
            <p style='color: #ffffff;'>Confidence: {confidence:.1%}</p>
        </div>
        """


def display_sources(sources: List[Dict[str, Any]]):
    """Display sources in a formatted way."""
    if not sources:
        st.info("No sources available")
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        st.markdown(f"""
        <div class='source-card'>
            <h4 style='color: #ffffff;'>Source {i}: {source['source']}</h4>
            <p style='color: #ffffff;'><strong>Type:</strong> {source['type']}</p>
            <p style='color: #ffffff;'><strong>Similarity:</strong> {source['similarity']:.1%}</p>
            {f"<p style='color: #ffffff;'><strong>Path:</strong> <code>{source.get('path', 'N/A')}</code></p>" if 'path' in source else ""}
        </div>
        """, unsafe_allow_html=True)


def add_border_to_image(img: Image.Image, border_color='#333333', border_width=3):
    """Add border to help white images stand out."""
    return ImageOps.expand(img, border=border_width, fill=border_color)


def main():
    """Main application."""
    
    # Header
    st.markdown("<div class='main-header'>🤖 Multimodal Multi-Agent RAG System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Powered by LangGraph, CLIP, and Cerebras LLaMA 70B</div>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        if st.button("🚀 Initialize System", use_container_width=True):
            initialize_system()
        
        st.divider()
        
        st.subheader("📊 System Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "LLaMA")
            st.metric("Top-K", settings.TOP_K_RESULTS)
        with col2:
            st.metric("Temp", settings.TEMPERATURE)
            st.metric("Threshold", f"{settings.SIMILARITY_THRESHOLD:.1%}")
        
        st.divider()
        
        # Show existing files in sidebar
        st.subheader("📁 Database Files")
        existing = get_existing_files()
        
        if existing['sources']:
            st.success(f"✓ {len(existing['sources'])} document(s)")
            st.metric("Total Images", existing['total_images'])
            
            with st.expander("📄 View Documents", expanded=False):
                for source, images in existing['sources'].items():
                    st.text(f"📄 {source}")
                    st.caption(f"   {len(images)} image(s)")
        else:
            st.info("No documents yet")
            st.caption("Upload in Documents tab")
        
        if existing['text_count'] > 0 or existing['image_count'] > 0:
            with st.expander("📊 Database Stats"):
                st.metric("Text Chunks", existing['text_count'])
                st.metric("Image Entries", existing['image_count'])
        
        st.divider()
        
        st.subheader("🎨 Features")
        st.success("✅ CLIP Image Search")
        st.success("✅ Local Embeddings")
        st.caption("Sentence Transformers + CLIP")
        
        st.divider()
        
        if st.button("🗑️ Reset System", type="secondary", use_container_width=True):
            if st.session_state.rag_system:
                with st.spinner("Resetting..."):
                    st.session_state.rag_system.reset()
                    st.session_state.ingested_files = []
                    st.session_state.chat_history = []
                    st.session_state.last_result = None
                    st.session_state.image_search_results = None
                    st.session_state.existing_files_loaded = False
                    st.success("✅ Reset complete!")
                    st.rerun()
        
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.session_state.image_search_results = None
            st.rerun()
        
        st.divider()
        
        with st.expander("❓ Help"):
            st.markdown("""
            **Text Search:**
            1. Ask questions in chat
            2. View retrieved images
            
            **Image Search:**
            1. Go to Image Search tab
            2. Upload query image
            3. Click Search
            4. View similar images
            
            **Features:**
            - CLIP visual similarity
            - Text + Image retrieval
            - Hallucination detection
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🖼️ Image Search", "📁 Documents", "📖 About"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat with Your Documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class='chat-message-user'>
                    <strong style='color: #90ee90;'>You:</strong><br>
                    <span style='color: #ffffff;'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message-assistant'>
                    <strong style='color: #90ee90;'>🤖 Assistant:</strong><br>
                    <span style='color: #ffffff;'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Ask a question:",
            placeholder="What would you like to know?",
            height=100,
            key="query_input"
        )
        
        if st.button("🚀 Send", type="primary", use_container_width=True):
            if st.session_state.rag_system is None:
                st.error("⚠️ Please initialize the system first!")
            elif query:
                with st.spinner("🔍 Processing..."):
                    try:
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': query
                        })
                        
                        result = st.session_state.rag_system.query(query)
                        
                        if isinstance(result, dict) and 'final_answer' in result:
                            st.session_state.last_result = result
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': result['final_answer']
                            })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display results
        if st.session_state.last_result:
            st.divider()
            
            # Display retrieved images
            retrieved_images = st.session_state.last_result.get('retrieved_images', [])
            
            if retrieved_images:
                st.subheader("🖼️ Retrieved Images")
                st.caption("Images displayed with borders for better visibility")
                
                cols = st.columns(min(len(retrieved_images), 2))
                for idx, img_data in enumerate(retrieved_images):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        img_path = img_data.get('path', '')
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                img_with_border = add_border_to_image(img)
                                
                                st.image(img_with_border, 
                                        caption=f"[Image {idx+1}] | Similarity: {img_data.get('similarity', 0):.1%} | "
                                               f"Page {img_data.get('metadata', {}).get('page', 'N/A')}",
                                        use_container_width=True)
                                
                                with st.expander(f"📍 Image {idx+1} Details"):
                                    st.text(f"Source: {img_data.get('metadata', {}).get('source_doc', 'Unknown')}")
                                    st.text(f"Page: {img_data.get('metadata', {}).get('page', 'N/A')}")
                                    st.text(f"Path: {img_path}")
                                    st.text(f"Similarity: {img_data.get('similarity', 0):.3f}")
                                
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                        else:
                            st.warning(f"Image not found: {img_path}")
            
            # Metrics and verification
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(
                    format_verification_status(st.session_state.last_result),
                    unsafe_allow_html=True
                )
                
                st.subheader("📊 Metrics")
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Texts", len(st.session_state.last_result.get('retrieved_texts', [])))
                with mcol2:
                    st.metric("Images", len(retrieved_images))
                with mcol3:
                    st.metric("Confidence", f"{st.session_state.last_result.get('confidence_score', 0):.1%}")
            
            with col2:
                display_sources(st.session_state.last_result.get('sources', []))
            
            with st.expander("🔬 Details"):
                for i, text in enumerate(st.session_state.last_result.get('retrieved_texts', [])[:5], 1):
                    st.markdown(f"""
                    **Chunk {i}** (Sim: {text.get('similarity', 0):.1%})  
                    Source: `{text.get('metadata', {}).get('source', 'unknown')}`  
                    {text.get('content', '')[:300]}...
                    """)
                    st.divider()
    
    # Tab 2: Image Search
    with tab2:
        st.header("🔍 CLIP Image Search")
        st.caption("Upload an image to find visually similar images in your database")
        
        if st.session_state.rag_system is None:
            st.warning("⚠️ Please initialize the system first!")
        elif not hasattr(st.session_state.rag_system, 'image_search') or \
             not st.session_state.rag_system.image_search or \
             not st.session_state.rag_system.image_search.available:
            st.error("❌ CLIP image search not available!")
            st.info("Install dependencies: `pip install transformers torch`")
        else:
            query_image = st.file_uploader(
                "Upload Query Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to find similar images in the database"
            )
            
            if query_image:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Query Image")
                    img = Image.open(query_image)
                    img_with_border = add_border_to_image(img, border_color='blue', border_width=5)
                    st.image(img_with_border, use_container_width=True)
                
                with col2:
                    st.subheader("Search Settings")
                    top_k = st.slider("Number of results", 1, 10, 5)
                    
                    if st.button("🔍 Search Similar Images", type="primary", use_container_width=True):
                        with st.spinner("🎨 Searching with CLIP..."):
                            try:
                                temp_dir = Path(settings.UPLOAD_DIR) / "temp"
                                temp_dir.mkdir(parents=True, exist_ok=True)
                                temp_path = temp_dir / query_image.name
                                
                                with open(temp_path, 'wb') as f:
                                    f.write(query_image.getbuffer())
                                
                                results = st.session_state.rag_system.search_by_image(str(temp_path), top_k=top_k)
                                st.session_state.image_search_results = results
                                
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                                
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            
            if st.session_state.image_search_results:
                st.divider()
                st.subheader("🎯 Search Results")
                
                results = st.session_state.image_search_results
                
                if results:
                    cols = st.columns(min(len(results), 3))
                    
                    for idx, result in enumerate(results):
                        col_idx = idx % 3
                        
                        with cols[col_idx]:
                            if os.path.exists(result['path']):
                                try:
                                    img = Image.open(result['path'])
                                    
                                    if result['similarity'] > 0.8:
                                        border_color = 'green'
                                        badge = '✅'
                                    elif result['similarity'] > 0.6:
                                        border_color = 'orange'
                                        badge = '⚠️'
                                    else:
                                        border_color = 'red'
                                        badge = '❌'
                                    
                                    img_with_border = add_border_to_image(img, border_color=border_color, border_width=4)
                                    
                                    st.image(img_with_border, use_container_width=True)
                                    st.caption(f"{badge} Match {idx+1}: {result['similarity']:.1%}")
                                    
                                    with st.expander(f"Details"):
                                        st.text(f"File: {result['filename']}")
                                        st.text(f"Path: {result['path']}")
                                        st.text(f"Similarity: {result['similarity']:.3f}")
                                    
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            else:
                                st.warning("Image not found")
                else:
                    st.info("No results found")
    
    # Tab 3: Document Management
    with tab3:
        st.header("Document Management")
        
        # Show existing files at top
        existing = get_existing_files()
        if existing['sources']:
            st.info(f"📁 Currently indexed: {len(existing['sources'])} document(s) with {existing['total_images']} image(s)")
            with st.expander("View Existing Documents"):
                for source, images in existing['sources'].items():
                    st.markdown(f"**📄 {source}**")
                    st.caption(f"Contains {len(images)} extracted image(s)")
                    st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'doc', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            if st.button("📥 Ingest", type="primary", disabled=not uploaded_files, use_container_width=True):
                if st.session_state.rag_system is None:
                    st.error("⚠️ Initialize first!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        temp_dir = Path(settings.UPLOAD_DIR) / "temp"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_paths = []
                        for i, f in enumerate(uploaded_files):
                            fp = temp_dir / f.name
                            with open(fp, 'wb') as file:
                                file.write(f.getbuffer())
                            file_paths.append(str(fp))
                            
                            progress_bar.progress((i+1)/len(uploaded_files)*0.5)
                            status_text.text(f"Saving {f.name}...")
                        
                        status_text.text("Ingesting...")
                        st.session_state.rag_system.ingest_documents(file_paths)
                        
                        for f in uploaded_files:
                            if f.name not in st.session_state.ingested_files:
                                st.session_state.ingested_files.append(f.name)
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success(f"✅ Ingested {len(uploaded_files)} files!")
                        
                        for fp in file_paths:
                            try:
                                os.remove(fp)
                            except:
                                pass
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.subheader("📋 Session Files")
            if st.session_state.ingested_files:
                for i, f in enumerate(st.session_state.ingested_files, 1):
                    st.text(f"{i}. {f}")
            else:
                st.info("None this session")
    
    # Tab 4: About
    with tab4:
        st.header("About This System")
        st.markdown(f"""
        ### 🎯 Features
        
        **Multi-Agent RAG:**
        - 🔍 Retrieval Agent - Finds relevant content
        - ✅ Verification Agent - Detects hallucinations
        - 💬 Response Agent - Generates answers
        
        **CLIP Image Search:**
        - 🎨 Visual similarity matching
        - 🖼️ Find similar diagrams/charts
        - 🔗 Cross-modal text-image search
        
        **Safety:**
        - Hallucination detection
        - Confidence scoring
        - Source citation
        
        ### 🚀 How to Use
        
        **Text Search:**
        1. Initialize system
        2. Upload & ingest documents
        3. Ask questions in Chat tab
        4. View retrieved images automatically
        
        **Image Search:**
        1. Go to Image Search tab
        2. Upload query image
        3. Click Search
        4. View similar images with scores
        
        ### ⚙️ Current Settings
        ```
        Model: {settings.MODEL_NAME}
        Similarity Threshold: {settings.SIMILARITY_THRESHOLD}
        Top-K Results: {settings.TOP_K_RESULTS}
        CLIP: Enabled
        ```
        
        ### 📊 Technology Stack
        - **LangGraph** - Multi-agent orchestration
        - **CLIP** - Multimodal embeddings
        - **Cerebras** - Fast LLM inference
        - **ChromaDB** - Vector database
        - **Streamlit** - Web interface
        """)
    
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #888;'>Multimodal RAG with CLIP | Powered by Cerebras</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
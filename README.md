# 🤖 Multimodal Multi-Agent RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system with **CLIP-powered multimodal search**, **multi-agent verification**, and **hallucination detection**. Built with LangGraph, Cerebras LLaMA 70B, and Streamlit.

<div align="center">

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

</div>

---

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Agent Architecture**: 3 specialized agents (Retrieval → Verification → Response)
- **Multimodal RAG**: Search across both text and images using CLIP embeddings
- **Visual Similarity Search**: Find diagrams by uploading query images
- **Hallucination Detection**: AI-powered verification with confidence scoring
- **Document Processing**: Automatic extraction of text and images from PDFs, DOCX, TXT

### 🎨 CLIP Integration
- **Cross-Modal Search**: Text queries find visually relevant images
- **Image-to-Image Search**: Upload an image to find similar diagrams
- **Semantic Understanding**: Goes beyond keyword matching
- **True Visual Embeddings**: Processes actual image pixels, not descriptions

### 🛡️ Safety & Quality
- **Verification Agent**: Validates responses against retrieved sources
- **Confidence Scoring**: 0-100% confidence for each answer
- **Source Citation**: Automatic attribution to source documents
- **Duplicate Filtering**: Removes inverted and duplicate images

### 🖥️ Interfaces
- **Streamlit Web UI**: Beautiful, interactive chat interface
- **CLI**: Command-line interface for automation
- **Docker Support**: One-command deployment anywhere

---

## 🏗️ Architecture

### Multi-Agent Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────────────┐
│  INPUT NODE                                     │
│  - Parse query                                  │
│  - Detect query type (text/image)               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  RETRIEVAL AGENT                                │
│  - CLIP text encoder → query embedding          │
│  - Search vector DB (ChromaDB)                  │
│  - Retrieve: Top-K texts + images               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  VERIFICATION AGENT                             │
│  - Compare answer vs sources                    │
│  - Detect hallucinations                        │
│  - Calculate confidence score                   │
│  - Status: verified/unverified/hallucination    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  RESPONSE AGENT                                 │
│  - Generate answer (Cerebras LLaMA 70B)         │
│  - Include image references                     │
│  - Add source citations                         │
│  - Format final response                        │
└─────────────────────────────────────────────────┘
    ↓
Final Answer + Images + Sources + Confidence
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | LangGraph |
| **LLM** | Cerebras LLaMA 70B |
| **Embeddings** | CLIP (ViT-Large) + Sentence Transformers |
| **Vector DB** | ChromaDB |
| **Document Processing** | PyMuPDF, python-docx |
| **Web UI** | Streamlit |
| **Containerization** | Docker |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Cerebras API Key ([Get one here](https://cerebras.ai/))
- Docker (optional, for containerized deployment)

### Option 1: Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

# 2. Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your CEREBRAS_API_KEY

# 5. Run Streamlit UI
streamlit run app_streamlit.py
```

Open browser: **http://localhost:8501**

### Option 2: Docker Deployment

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env and add your API key

# 2. Run with Docker Compose
docker-compose up

# Or use the simple version (no build required)
docker-compose -f docker-compose.simple.yml up
```

Open browser: **http://localhost:8501**

---

## 📖 Usage Guide

### Web Interface (Streamlit)

#### 1. Upload Documents
```
Documents Tab → Upload PDF/DOCX → Click "Ingest"
```

#### 2. Text Search
```
Chat Tab → Type: "Explain transformer architecture" → Send
```
Returns: Text answer + relevant images + sources

#### 3. Image Search
```
Image Search Tab → Upload query image → Click "Search Similar Images"
```
Returns: Visually similar images ranked by CLIP similarity

### Command Line Interface

#### Ingest Documents
```bash
python main.py --mode ingest --files "path/to/document.pdf"
```

#### Text Query
```bash
python main.py --mode query --query "What is attention mechanism?"
```

#### Image Search
```bash
python main.py --mode image_search --image "path/to/diagram.png"
```

#### Interactive Mode
```bash
python main.py --mode interactive
> transformer architecture
> image diagram.png
> exit
```

#### Reset Database
```bash
python main.py --mode reset
```

---

## 📁 Project Structure

```
Multimodal_Multi-agent_RAG/
│
├── agents/                      # Multi-agent components
│   ├── retrieval_agent.py      # Retrieves relevant content
│   ├── verification_agent.py    # Validates responses
│   └── response_agent.py        # Generates final answer
│
├── utils/                       # Core utilities
│   ├── document_processor.py   # Extract text + images from docs
│   ├── vector_store.py         # ChromaDB vector database
│   ├── multimodal_embeddings.py # CLIP embeddings
│   └── image_search.py         # Visual similarity search
│
├── tools/                       # Agent tools
│   └── retrieval_tools.py      # Search and fetch tools
│
├── data/                        # Data directory
│   ├── uploads/                # Uploaded documents
│   │   └── extracted_images/   # Extracted diagrams
│   └── chroma_db/              # Vector database
│
├── graph.py                     # LangGraph workflow
├── nodes.py                     # Graph node definitions
├── state.py                     # State management
├── config.py                    # Configuration
├── main.py                      # CLI entrypoint
├── app_streamlit.py            # Streamlit web UI
│
├── Dockerfile                   # Docker image definition
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required
CEREBRAS_API_KEY=your_api_key_here

# Model Settings
MODEL_NAME=llama-3.3-70b
TEMPERATURE=0.0
MAX_TOKENS=1000

# Retrieval Settings
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3  

# Paths
CHROMA_DB_PATH=./data/chroma_db
UPLOAD_DIR=./data/uploads

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_IMAGE_SIZE=800,800
```

### Model Selection

**CLIP Models** (in `utils/multimodal_embeddings.py`):
- `openai/clip-vit-base-patch32` - Fast, 150MB (recommended)
- `openai/clip-vit-large-patch14` - Accurate, 500MB (slower)

**LLM Models**:
- `llama3.1-70b` - Best quality (default)
- `llama3.1-8b` - Faster, lower cost

---

## 🎯 Key Features Explained

### 1. CLIP Multimodal Search

**Problem:** Traditional RAG only matches keywords. "transformer architecture" won't find diagrams labeled "encoder-decoder model".

**Solution:** CLIP creates a shared embedding space:
```python
Text: "transformer architecture" → [0.23, 0.87, -0.45, ...]
Image: [actual diagram pixels]  → [0.21, 0.89, -0.43, ...]
Similarity: 0.94 ✅ (They represent the same thing!)
```

**Result:** Text queries find semantically similar images, even with different labels!

### 2. Multi-Agent Verification

**Problem:** LLMs hallucinate - they make up facts not in the sources.

**Solution:** 3-agent pipeline:
1. **Retrieval Agent**: Finds relevant content
2. **Verification Agent**: Checks if answer is supported by sources
3. **Response Agent**: Generates verified answer with citations

**Result:** Confidence scores and hallucination warnings!

### 3. Image Deduplication

**Problem:** PDFs contain duplicate images (normal + inverted versions).

**Solution:** Intelligent filtering:
- Skip dark images (mean brightness < 50)
- Skip blank images (mean brightness > 250)  
- Skip duplicates (image similarity)
- Skip tiny images (< 100x100)

**Result:** Only clean, unique, readable images!

---

## 🔬 Advanced Usage

### Custom Embedding Model

Edit `utils/vector_store.py`:
```python
def _initialize_embeddings(self):
    clip = MultimodalEmbeddings(model_name="openai/clip-vit-base-patch32")
    # Change to: "openai/clip-vit-large-patch14" for higher accuracy
```

### Adjust Similarity Threshold

Lower threshold = more results (less strict):
```python
# In .env
SIMILARITY_THRESHOLD=0.1  # Permissive (recommended for CLIP)
SIMILARITY_THRESHOLD=0.3  # Balanced
SIMILARITY_THRESHOLD=0.5  # Strict
```

### Batch Document Ingestion

```python
from main import MultimodalRAGSystem

system = MultimodalRAGSystem()
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
system.ingest_documents(documents)
```

---

## 🐛 Troubleshooting

### No Images Retrieved

**Symptom:** `Retrieved 5 texts and 0 images`

**Solution:**
```bash
# 1. Check threshold
# Lower SIMILARITY_THRESHOLD in .env to 0.1

# 2. Verify CLIP embeddings
python -c "from utils.multimodal_embeddings import MultimodalEmbeddings; print('✓ CLIP OK')"

# 3. Re-ingest documents
python main.py --mode reset
python main.py --mode ingest --files "your_doc.pdf"
```

### Docker Build Fails

**Symptom:** `error writing manifest blob`

**Solution:**
```bash
# Use simple compose (no build)
docker-compose -f docker-compose.simple.yml up

# Or increase Docker resources:
# Docker Desktop → Settings → Resources → Disk: 100GB, Memory: 8GB
```

### CLIP Loading Slow

**First run:** CLIP downloads 500MB model (one-time)

**Speed up:** Use smaller model:
```python
# In utils/multimodal_embeddings.py
MultimodalEmbeddings(model_name="openai/clip-vit-base-patch32")  # 150MB
```

---

## 📊 Performance

### Benchmarks (on CPU)

| Operation | Time | Notes |
|-----------|------|-------|
| **Document Ingestion** | ~30s per 10-page PDF | Includes image extraction + embedding |
| **Text Query** | 2-5s | Retrieval (0.5s) + LLM (2-4s) |
| **Image Search** | 1-2s | CLIP embedding + similarity search |
| **CLIP Model Load** | 10-20s | One-time on startup |

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4 GB | 8 GB |
| **Disk** | 5 GB | 20 GB |
| **Python** | 3.11+ | 3.11 |
| **GPU** | Not required | Optional (10x faster CLIP) |

---

## 🔐 Security Best Practices

1. **Never commit `.env`** - Contains API keys
2. **Add to `.gitignore`:**
   ```
   .env
   data/
   myenv/
   __pycache__/
   ```
3. **Use environment-specific configs:**
   - `.env.development`
   - `.env.production`
4. **Rotate API keys** regularly
5. **Limit file upload sizes** in production

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangGraph** - Multi-agent orchestration framework
- **Cerebras** - Fast LLM inference
- **OpenAI CLIP** - Multimodal embeddings
- **ChromaDB** - Vector database
- **Streamlit** - Web interface framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multimodal-rag/discussions)
- **Email**: ahmedessamhamam@gmail.com

---

## 🗺️ Roadmap

- [ ] GPU acceleration for CLIP
- [ ] Support for more document formats (PPT, Excel)
- [ ] Multi-language support
- [ ] API endpoint (FastAPI)
- [ ] Ollama integration (local LLMs)
- [ ] Vector database alternatives (Pinecone, Weaviate)
- [ ] Advanced visualization of retrieval results
- [ ] Batch processing API

---

## 📈 Changelog

### v1.0.0 (2026-02-14)
- ✨ Initial release
- 🎨 CLIP multimodal search
- 🤖 Multi-agent verification
- 🖼️ Image-to-image search
- 🐳 Docker support
- 📱 Streamlit web UI

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Built with ❤️ using LangGraph, CLIP, and Cerebras

</div>

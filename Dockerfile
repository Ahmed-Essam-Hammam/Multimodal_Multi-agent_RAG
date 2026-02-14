# Multi-stage build for smaller image size
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages to a specific directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final stage - smaller image
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads/extracted_images data/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first start works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY brain ./brain
COPY scripts ./scripts
COPY app.py .

ENV BRAIN_DATA_DIR=/data \
    OLLAMA_URL=http://host.docker.internal:11434
VOLUME /data
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]

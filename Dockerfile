FROM python:3.12-slim

WORKDIR /app

# System dependencies for OpenCV, InsightFace, video processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir insightface onnxruntime

# Copy backend code
COPY backend/ ./
COPY dashboard/ ../dashboard/

# Hugging Face requires writable cache locations and the user runs as non-root
RUN mkdir -p uploads/missing_persons uploads/videos models /tmp/.cache && \
    chmod -R 777 uploads models /tmp/.cache

# Hugging Face uses port 7860
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HOME=/tmp
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

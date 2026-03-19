FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV + insightface
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir insightface onnxruntime

# Copy backend code
COPY backend/*.py ./
COPY backend/modules/ ./modules/
COPY backend/routers/ ./routers/
COPY backend/models/ ./models/

# Copy dashboard for serving
COPY dashboard/ ./dashboard/

# Create directories
RUN mkdir -p uploads/missing_persons uploads/videos

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

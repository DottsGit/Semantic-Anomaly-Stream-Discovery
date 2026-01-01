FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pre-download DINOv2 model (optional - will download on first run if not present)
# RUN python -c "from transformers import AutoModel, AutoImageProcessor; AutoModel.from_pretrained('facebook/dinov2-base'); AutoImageProcessor.from_pretrained('facebook/dinov2-base')"

# Expose ports
EXPOSE 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.main"]

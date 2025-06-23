FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmagic1 \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Copy application code
COPY app/ /app/

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
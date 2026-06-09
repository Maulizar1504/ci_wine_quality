FROM python:3.12.7-slim

LABEL maintainer="Maulizar"
LABEL description="Wine Quality ML Inference API"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY Membangun_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Membangun_model/wine_quality_preprocessing.py .
COPY "Monitoring dan Logging/7.Inference.py" ./Inference.py

# Model files (mounted as volumes in docker-compose)
# COPY Membangun_model/best_tuned_model.pkl .
# COPY Membangun_model/scaler.pkl .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["python", "Inference.py"]

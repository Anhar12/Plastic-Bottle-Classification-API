FROM python:3.10-slim

# Install system libs required by OpenCV + basic runtime deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Railway uses $PORT, default fallback 8080
ENV PORT=8080

# Gunicorn entrypoint
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 1 -b 0.0.0.0:${PORT} app:app --timeout 120"]

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Create data directories
RUN mkdir -p /app/data/songs /app/config

# Set Python path
ENV PYTHONPATH=/app/backend

# Expose port
EXPOSE 8000

# Copy and make entrypoint executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Run the application
CMD ["/app/entrypoint.sh"]

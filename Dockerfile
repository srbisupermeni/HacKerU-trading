FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for numerical computation and trading libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory with proper permissions
RUN mkdir -p /app/bot/logs && chmod 755 /app/bot/logs

# Environment configuration
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check: verify bot is still running
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD ps aux | grep -v grep | grep "python main.py" > /dev/null || exit 1

# Run main trading bot
CMD ["python", "main.py"]

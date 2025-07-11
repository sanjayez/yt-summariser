#!/bin/bash

# Single Celery Worker with High Concurrency for macOS
# Alternative approach using threads instead of processes

echo "🚀 Starting YouTube Summarizer Concurrent Celery Worker..."

# Set macOS-specific environment variables
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=4
export PYTHONDONTWRITEBYTECODE=1

# Kill any existing Celery workers
echo "🧹 Cleaning up existing workers..."
pkill -f "celery.*worker" || true
sleep 2

# Check Docker and Redis connection
echo "🔍 Checking Docker and Redis connection..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Redis container is running
if ! docker ps | grep -q "redis-server"; then
    echo "❌ Redis container is not running. Please start with:"
    echo "   docker-compose up -d redis"
    exit 1
fi

# Test Redis connection via Docker
if ! docker exec redis-server redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not responding. Please check Redis container:"
    echo "   docker-compose logs redis"
    exit 1
fi

echo "✅ Redis is running in Docker"

# Start single worker with multiple threads
echo "🎬 Starting Celery worker with thread-based concurrency..."

celery -A yt_summariser worker \
    --loglevel=info \
    --pool=threads \
    --concurrency=4 \
    --prefetch-multiplier=1 \
    --max-tasks-per-child=10 \
    --max-memory-per-child=500000 \
    --without-gossip \
    --without-mingle \
    --time-limit=2100 \
    --soft-time-limit=1800

echo "👋 Celery worker stopped"
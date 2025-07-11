#!/bin/bash

# Parallel Celery Worker Startup Script for macOS
# Optimized for parallel video processing

echo "🚀 Starting YouTube Summarizer Parallel Celery Workers..."

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

# Number of workers (adjust based on your system)
WORKER_COUNT=${1:-3}  # Default to 3 workers, can be overridden

echo "🎬 Starting $WORKER_COUNT Celery workers for parallel processing..."

# Start multiple workers in background
for i in $(seq 1 $WORKER_COUNT); do
    echo "Starting worker $i..."
    celery -A yt_summariser worker \
        --loglevel=info \
        --hostname=worker$i@%h \
        --concurrency=1 \
        --prefetch-multiplier=1 \
        --max-tasks-per-child=5 \
        --max-memory-per-child=300000 \
        --without-gossip \
        --without-mingle \
        --time-limit=2100 \
        --soft-time-limit=1800 \
        --detach \
        --pidfile=/tmp/celery_worker$i.pid \
        --logfile=/tmp/celery_worker$i.log
    
    sleep 1  # Small delay between worker starts
done

echo "✅ Started $WORKER_COUNT parallel Celery workers"
echo "📊 Monitor workers with: celery -A yt_summariser inspect active"
echo "📋 Stop workers with: pkill -f 'celery.*worker'"
echo "📄 Worker logs: /tmp/celery_worker*.log"

# Optional: Start flower for monitoring (comment out if not needed)
# echo "🌸 Starting Celery Flower monitoring..."
# celery -A yt_summariser flower --port=5555 &

echo "🎉 Parallel processing ready!"
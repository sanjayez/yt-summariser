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

# Configuration
WORKER_COUNT=${1:-3}  # Default to 3 workers, can be overridden
START_FLOWER=${2:-"yes"}  # Default to starting Flower, pass "no" to disable
FLOWER_PORT=${3:-5555}   # Default Flower port

# Kill any existing Celery workers and Flower
echo "🧹 Cleaning up existing workers and Flower..."
pkill -f "celery.*worker" || true
pkill -f "celery.*flower" || true
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

echo "🎬 Starting $WORKER_COUNT Celery workers for parallel processing..."

# Start multiple workers in background
for i in $(seq 1 $WORKER_COUNT); do
    echo "Starting worker $i..."
    uv run celery -A yt_summariser worker \
        --loglevel=info \
        --hostname=worker$i@%h \
        --concurrency=4 \
        --prefetch-multiplier=1 \
        --max-tasks-per-child=20 \
        --max-memory-per-child=300000 \
        --time-limit=2100 \
        --soft-time-limit=1800 \
        --detach \
        --pidfile=/tmp/celery_worker$i.pid \
        --logfile=/tmp/celery_worker$i.log

    sleep 1  # Small delay between worker starts
done

echo "✅ Started $WORKER_COUNT parallel Celery workers"

# Start Flower monitoring if requested
if [ "$START_FLOWER" = "yes" ]; then
    echo "🌸 Starting Celery Flower monitoring..."

    # Check if this is likely a production environment
    if [ -n "$DJANGO_ENV" ] && [ "$DJANGO_ENV" = "production" ]; then
        # Production Flower with authentication and persistence
        echo "🔒 Starting Flower in PRODUCTION mode with authentication..."
        uv run celery -A yt_summariser flower \
            --port=$FLOWER_PORT \
            --address=127.0.0.1 \
            --max_tasks=10000 \
            --db=flower_prod.db \
            --persistent=True \
            --basic_auth=admin:secure_flower_2024 \
            --loglevel=warning \
            --detach \
            --pidfile=/tmp/flower.pid \
            --logfile=/tmp/flower.log
    else
        # Development Flower - more open and verbose
        echo "🛠️ Starting Flower in DEVELOPMENT mode..."
        uv run celery -A yt_summariser flower \
            --port=$FLOWER_PORT \
            --address=127.0.0.1 \
            --max_tasks=5000 \
            --db=flower_dev.db \
            --persistent=True \
            --loglevel=info \
            --detach \
            --pidfile=/tmp/flower.pid \
            --logfile=/tmp/flower.log
    fi

    # Wait a moment for Flower to start
    sleep 3

    # Check if Flower started successfully
    if curl -s http://localhost:$FLOWER_PORT > /dev/null 2>&1; then
        echo "✅ Flower started successfully!"
        echo "🌸 Monitor at: http://localhost:$FLOWER_PORT"
        if [ -n "$DJANGO_ENV" ] && [ "$DJANGO_ENV" = "production" ]; then
            echo "🔐 Use credentials: admin / secure_flower_2024"
        fi
    else
        echo "⚠️ Flower may have failed to start. Check /tmp/flower.log"
    fi
else
    echo "⏭️ Skipping Flower startup (disabled)"
fi

echo ""
echo "📊 Monitor workers with: uv run celery -A yt_summariser inspect active"
echo "📋 Stop all with: pkill -f 'celery.*worker'; pkill -f 'celery.*flower'"
echo "📄 Worker logs: /tmp/celery_worker*.log"
if [ "$START_FLOWER" = "yes" ]; then
    echo "🌸 Flower log: /tmp/flower.log"
fi

echo "🎉 Parallel processing ready!"

#!/bin/bash

# Robust Celery Startup Script for macOS
# Enhanced for long-running video processing tasks

echo "🚀 Starting YouTube Summarizer Celery Worker..."

# Set macOS-specific environment variables
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# macOS system optimization
export MALLOC_ARENA_MAX=4
export PYTHONDONTWRITEBYTECODE=1

# Kill any existing Celery workers
echo "🧹 Cleaning up existing workers..."
pkill -f "celery.*worker" || true
sleep 2

celery -A yt_summariser worker --loglevel=info

# # Ensure Redis is running
# echo "🔍 Checking Redis connection..."
# redis-cli ping > /dev/null 2>&1
# if [ $? -ne 0 ]; then
#     echo "❌ Redis is not running. Please start Redis first:"
#     echo "   brew services start redis"
#     exit 1
# fi

# echo "✅ Redis is running"

# Start Celery worker with enhanced settings for macOS
# echo "🎬 Starting Celery worker with macOS optimizations..."

# celery -A yt_summariser worker \
#     --loglevel=info \
#     --pool=solo \
#     --concurrency=1 \
#     --prefetch-multiplier=1 \
#     --max-tasks-per-child=5 \
#     --max-memory-per-child=300000 \
#     --without-gossip \
#     --without-mingle \
#     --without-heartbeat \
#     --task-events \
#     --time-limit=2100 \
#     --soft-time-limit=1800

# echo "👋 Celery worker stopped" 
#!/bin/bash

# Enhanced server startup script for yt-summariser
# Supports both development and production modes

set -e  # Exit on any error

# Default configuration
MODE="development"
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE       Set server mode: development, production (default: development)"
            echo "  --host HOST       Set bind host (default: 127.0.0.1)"
            echo "  --port PORT       Set port (default: 8000)"
            echo "  --workers WORKERS Set number of workers for production (default: 1)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Development mode"
            echo "  $0 --mode production --workers 4     # Production mode with 4 workers"
            echo "  $0 --host 0.0.0.0 --port 8080       # Bind to all interfaces on port 8080"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting yt-summariser server..."
echo "   Mode: $MODE"
echo "   Host: $HOST"
echo "   Port: $PORT"

case $MODE in
    "development")
        echo "   Using: Daphne (development server)"
        echo ""
        exec daphne -b "$HOST" -p "$PORT" yt_summariser.asgi:application
        ;;
    "production")
        echo "   Using: Gunicorn with Uvicorn workers"
        echo "   Workers: $WORKERS"
        echo ""
        exec gunicorn yt_summariser.asgi:application \
            -k uvicorn.workers.UvicornWorker \
            -w "$WORKERS" \
            -b "$HOST:$PORT" \
            --worker-connections 1000 \
            --max-requests 10000 \
            --max-requests-jitter 1000 \
            --preload \
            --access-logfile - \
            --error-logfile -
        ;;
    *)
        echo "❌ Error: Unknown mode '$MODE'. Use 'development' or 'production'"
        exit 1
        ;;
esac
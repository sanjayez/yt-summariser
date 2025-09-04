#!/bin/bash

# Log Management Utility for YT Summariser
# Provides commands for viewing, cleaning, and managing centralized logs

set -e

LOGS_DIR="logs"
ARCHIVE_DIR="$LOGS_DIR/archived"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "🗂️ Log Management Utility for YT Summariser"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  status          Show current log file sizes and locations"
    echo "  tail [SERVICE]  Tail logs (django|celery|monitoring|all)"
    echo "  clean [DAYS]    Clean logs older than DAYS (default: 30)"
    echo "  archive         Manually archive current logs"
    echo "  search PATTERN  Search across all logs for pattern"
    echo "  monitor         Real-time monitoring of all services"
    echo ""
    echo "Examples:"
    echo "  $0 status                    # Show log status"
    echo "  $0 tail celery              # Tail Celery worker logs"
    echo "  $0 tail all                 # Tail all logs"
    echo "  $0 clean 7                  # Clean logs older than 7 days"
    echo "  $0 search 'LLM enhancement' # Search for LLM logs"
    echo "  $0 monitor                  # Monitor all services"
}

status() {
    echo "📊 Log Status Report"
    echo "===================="
    echo ""

    if [ -d "$LOGS_DIR" ]; then
        echo -e "${GREEN}📁 Log Directory Structure:${NC}"
        tree "$LOGS_DIR" 2>/dev/null || find "$LOGS_DIR" -type f -exec ls -lh {} \;
        echo ""

        echo -e "${BLUE}💾 Disk Usage by Service:${NC}"
        for service_dir in "$LOGS_DIR"/*; do
            if [ -d "$service_dir" ]; then
                service_name=$(basename "$service_dir")
                size=$(du -sh "$service_dir" 2>/dev/null | cut -f1)
                echo "  $service_name: $size"
            fi
        done
        echo ""

        echo -e "${YELLOW}📈 Recent Activity (last 24h):${NC}"
        find "$LOGS_DIR" -name "*.log" -o -name "*.jsonl" | while read -r logfile; do
            if [ -f "$logfile" ]; then
                size=$(wc -l < "$logfile" 2>/dev/null || echo "0")
                modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$logfile" 2>/dev/null || echo "unknown")
                echo "  $(basename "$logfile"): $size lines (modified: $modified)"
            fi
        done
    else
        echo -e "${RED}❌ Logs directory not found. Run with Django server to create.${NC}"
    fi
}

tail_logs() {
    local service="$1"

    case "$service" in
        "django"|"application")
            echo "📱 Tailing Django logs..."
            tail -f "$LOGS_DIR"/application/*.log 2>/dev/null || echo "No Django logs found"
            ;;
        "celery")
            echo "⚙️ Tailing Celery worker logs..."
            tail -f "$LOGS_DIR"/celery/*.log 2>/dev/null || echo "No Celery logs found"
            ;;
        "monitoring")
            echo "📊 Tailing monitoring logs..."
            tail -f "$LOGS_DIR"/monitoring/*.log 2>/dev/null || echo "No monitoring logs found"
            ;;
        "all"|"")
            echo "🔍 Tailing all service logs..."
            tail -f "$LOGS_DIR"/*/*.log "$LOGS_DIR"/*/*.jsonl 2>/dev/null || echo "No logs found"
            ;;
        *)
            echo -e "${RED}❌ Unknown service: $service${NC}"
            echo "Available: django, celery, monitoring, all"
            exit 1
            ;;
    esac
}

clean_logs() {
    local days="${1:-30}"

    echo "🧹 Cleaning logs older than $days days..."

    # Find and clean old log files
    local cleaned=0
    find "$LOGS_DIR" -name "*.log*" -o -name "*.jsonl*" | while read -r logfile; do
        if [ -f "$logfile" ] && [ "$(find "$logfile" -mtime +$days)" ]; then
            echo "  Removing: $(basename "$logfile")"
            rm -f "$logfile"
            ((cleaned++))
        fi
    done

    echo "✅ Cleaned $cleaned old log files"

    # Compress archived logs older than 7 days
    find "$ARCHIVE_DIR" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    echo "🗜️ Compressed old archived logs"
}

archive_current() {
    echo "📦 Archiving current logs..."

    local timestamp=$(date +%Y-%m-%d_%H-%M-%S)

    # Archive current logs
    for service_dir in "$LOGS_DIR"/{application,celery,monitoring}; do
        if [ -d "$service_dir" ]; then
            service_name=$(basename "$service_dir")
            find "$service_dir" -name "*.log" -exec mv {} "$ARCHIVE_DIR/${service_name}-{}-$timestamp" \; 2>/dev/null || true
        fi
    done

    echo "✅ Current logs archived with timestamp: $timestamp"
}

search_logs() {
    local pattern="$1"

    if [ -z "$pattern" ]; then
        echo -e "${RED}❌ Please provide a search pattern${NC}"
        exit 1
    fi

    echo "🔍 Searching for: '$pattern'"
    echo "================================"

    # Search in current logs
    grep -r -n --color=always "$pattern" "$LOGS_DIR" 2>/dev/null || echo "No matches found in current logs"

    # Search in archived logs
    echo ""
    echo "📦 Archived logs:"
    grep -r -n --color=always "$pattern" "$ARCHIVE_DIR" 2>/dev/null || echo "No matches found in archived logs"
}

monitor_services() {
    echo "📊 Real-time Service Monitoring"
    echo "=============================="
    echo "Press Ctrl+C to stop"
    echo ""

    # Monitor all logs with service labels
    tail -f "$LOGS_DIR"/*/*.log "$LOGS_DIR"/*/*.jsonl 2>/dev/null | while IFS= read -r line; do
        timestamp=$(date '+%H:%M:%S')
        echo "[$timestamp] $line"
    done
}

# Main command dispatch
case "${1:-help}" in
    "status")
        status
        ;;
    "tail")
        tail_logs "$2"
        ;;
    "clean")
        clean_logs "$2"
        ;;
    "archive")
        archive_current
        ;;
    "search")
        search_logs "$2"
        ;;
    "monitor")
        monitor_services
        ;;
    "help"|"--help"|"-h")
        usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

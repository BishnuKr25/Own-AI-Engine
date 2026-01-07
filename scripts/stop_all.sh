#!/bin/bash
# Stop all services

echo "Stopping Sovereign AI Suite..."

# Stop processes
for pid_file in logs/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null; then
            echo "Stopping process $pid..."
            kill $pid
        fi
        rm "$pid_file"
    fi
done

echo "✓ All services stopped"
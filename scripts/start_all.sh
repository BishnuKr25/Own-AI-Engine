#!/bin/bash
# Production startup script

set -e

echo "Starting Sovereign AI Suite..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root"
   exit 1
fi

# Set environment
export SOVEREIGN_AI_HOME="/opt/sovereign-ai-suite"
cd $SOVEREIGN_AI_HOME

# Activate virtual environment
source venv/bin/activate

# Start MongoDB if not running
if ! pgrep -x "mongod" > /dev/null; then
    echo "Starting MongoDB..."
    sudo systemctl start mongod
fi

# Start Redis if using caching
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    sudo systemctl start redis-server
fi

# Start backend API
echo "Starting Backend API..."
nohup python -m backend.main > logs/backend.log 2>&1 &
echo $! > logs/backend.pid

# Wait for backend to be ready
echo "Waiting for backend..."
sleep 10

# Start admin portal
echo "Starting Admin Portal..."
nohup python admin/app.py > logs/admin.log 2>&1 &
echo $! > logs/admin.pid

# Start frontend
echo "Starting Frontend..."
nohup streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > logs/frontend.log 2>&1 &
echo $! > logs/frontend.pid

echo "✓ All services started"
echo ""
echo "Access points:"
echo "  Frontend: http://localhost:8501"
echo "  Admin: http://localhost:5000"
echo "  API: http://localhost:8000"
echo ""
echo "Default admin credentials:"
echo "  Username: admin"
echo "  Password: changeme123"
echo ""
echo "To stop all services, run: ./scripts/stop_all.sh"
#!/bin/bash
# Start the neural service with GPU acceleration

set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
PORT=50052
POLICY_WEIGHTS=""
VALUE_WEIGHTS=""
SHUTDOWN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --policy-weights=*)
            POLICY_WEIGHTS="--policy-weights=${1#*=}"
            shift
            ;;
        --value-weights=*)
            VALUE_WEIGHTS="--value-weights=${1#*=}"
            shift
            ;;
        --shutdown)
            SHUTDOWN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port=PORT] [--policy-weights=PATH] [--value-weights=PATH] [--shutdown]"
            exit 1
            ;;
    esac
done

# Create a PID file for easier service management
PID_FILE="python/neural_service_$PORT.pid"

# Handle shutdown if requested
if [ "$SHUTDOWN" = true ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping neural service (PID: $PID) on port $PORT..."
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            echo "Neural service stopped."
        else
            echo "Neural service is not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found for neural service on port $PORT."
    fi
    exit 0
fi

# Check if an existing service is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "Neural service is already running with PID $PID on port $PORT."
        echo "Use '$0 --shutdown' to stop it first, or use a different port."
        exit 0
    else
        echo "Removing stale PID file from previous run."
        rm -f "$PID_FILE"
    fi
fi

# Check if virtual environment exists
if [ ! -d "python/venv" ]; then
    echo "Python environment not found. Setting up..."
    ./python/setup_local_env.sh
fi

# Start the service
echo "Starting neural service on port $PORT..."
source python/venv/bin/activate

# Start service in the background and save PID
python python/neural_service.py --port $PORT $POLICY_WEIGHTS $VALUE_WEIGHTS &
SERVICE_PID=$!
echo $SERVICE_PID > $PID_FILE
echo "Neural service started with PID $SERVICE_PID"

# Wait a moment for the service to start
sleep 2

# Verify the service is running
if kill -0 $SERVICE_PID 2>/dev/null; then
    echo "Neural service is running successfully on port $PORT."
else
    echo "Failed to start neural service."
    rm -f $PID_FILE
    exit 1
fi 
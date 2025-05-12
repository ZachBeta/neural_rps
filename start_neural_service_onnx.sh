#!/bin/bash
# Start the ONNX neural service with GPU acceleration

set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
PORT=50053 # Default port for ONNX service
MODEL_PATH_ARG="python/output/rps_h256_value.model.onnx" # Default ONNX model path (H256)
SHUTDOWN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --model-path=*)
            MODEL_PATH_ARG="${1#*=}"
            shift
            ;;
        --shutdown)
            SHUTDOWN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port=PORT] [--model-path=PATH_TO_ONNX_MODEL] [--shutdown]"
            exit 1
            ;;
    esac
done

# Create a PID file for easier service management
PID_FILE="python/neural_service_onnx_$PORT.pid"

# Handle shutdown if requested
if [ "$SHUTDOWN" = true ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping ONNX neural service (PID: $PID) on port $PORT..."
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            echo "ONNX Neural service stopped."
        else
            echo "ONNX Neural service is not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found for ONNX neural service on port $PORT."
    fi
    exit 0
fi

# Check if an existing service is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "ONNX Neural service is already running with PID $PID on port $PORT."
        echo "Use '$0 --shutdown --port=$PORT' to stop it first, or use a different port."
        exit 0
    else
        echo "Removing stale PID file from previous run."
        rm -f "$PID_FILE"
    fi
fi

# Check if virtual environment exists
if [ ! -d "python/venv" ]; then # Adjusted to python/venv based on previous context
    echo "Python virtual environment not found in python/venv. Setting up..."
    # Assuming setup_local_env.sh is in the python directory and creates python/venv
    if [ -f "python/setup_local_env.sh" ]; then
        (cd python && ./setup_local_env.sh) # Run setup from within python dir
    else
        echo "Error: python/setup_local_env.sh not found! Cannot setup virtual environment."
        exit 1
    fi
fi

# Start the service
echo "Starting ONNX neural service on port $PORT with model $MODEL_PATH_ARG..."

# Activate virtual environment - ensure this path is correct for your setup
if [ -f "python/venv/bin/activate" ]; then
    source python/venv/bin/activate
else
    echo "Error: python/venv/bin/activate not found! Please ensure virtual env is set up correctly."
    exit 1
fi

# Start service in the background and save PID
# Ensure the python script itself is executable or called with python interpreter
python -u python/neural_service_onnx.py --port "$PORT" --model_path "$MODEL_PATH_ARG" &
SERVICE_PID=$!
echo $SERVICE_PID > "$PID_FILE"
echo "ONNX Neural service started with PID $SERVICE_PID"

# Wait a moment for the service to start
sleep 2

# Verify the service is running
if kill -0 $SERVICE_PID 2>/dev/null; then
    echo "ONNX Neural service is running successfully on port $PORT."
else
    echo "Failed to start ONNX neural service."
    rm -f "$PID_FILE"
    exit 1
fi 
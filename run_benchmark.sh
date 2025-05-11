#!/bin/bash
# Run the benchmark comparing CPU and GPU neural network performance

set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments first to get the port
PORT=50051
BATCH_SIZE=64
ITERATIONS=100
ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --addr=*)
            ADDR="${1#*=}"
            PORT=$(echo $ADDR | cut -d':' -f2)
            ARGS="$ARGS $1"
            shift
            ;;
        --batch-size=*)
            BATCH_SIZE="${1#*=}"
            ARGS="$ARGS $1"
            shift
            ;;
        --iterations=*)
            ITERATIONS="${1#*=}"
            ARGS="$ARGS $1"
            shift
            ;;
        --cpu-only)
            ARGS="$ARGS $1"
            shift
            ;;
        --gpu-only)
            ARGS="$ARGS $1"
            shift
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# Check if Python service is running
if ! nc -z localhost $PORT 2>/dev/null; then
    echo "Starting neural service..."
    
    # Check if virtual environment exists
    if [ ! -d "python/venv" ]; then
        echo "Python environment not found. Setting up..."
        ./python/setup_local_env.sh
    fi
    
    # Start the neural service in the background
    source python/venv/bin/activate
    python python/neural_service.py --port $PORT &
    NEURAL_SERVICE_PID=$!
    
    # Ensure we kill the service when the script exits
    trap "kill $NEURAL_SERVICE_PID 2>/dev/null || true" EXIT
    
    # Wait for service to start
    echo "Waiting for neural service to start..."
    for i in {1..10}; do
        if nc -z localhost $PORT 2>/dev/null; then
            echo "Neural service is running."
            break
        fi
        sleep 1
        if [ $i -eq 10 ]; then
            echo "Timed out waiting for neural service to start."
            exit 1
        fi
    done
else
    echo "Neural service is already running on port $PORT."
fi

# Run the benchmark
echo "Running benchmark with batch size $BATCH_SIZE and $ITERATIONS iterations..."
go run cmd/benchmark/main.go --batch-size=$BATCH_SIZE --iterations=$ITERATIONS $ARGS 
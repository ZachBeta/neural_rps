#!/bin/bash
# Start the neural service with GPU acceleration

set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
PORT=50051
POLICY_WEIGHTS=""
VALUE_WEIGHTS=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port=PORT] [--policy-weights=PATH] [--value-weights=PATH]"
            exit 1
            ;;
    esac
done

# Check if the service is already running on the specified port
if nc -z localhost $PORT 2>/dev/null || lsof -i :$PORT > /dev/null 2>&1; then
    echo "Neural service is already running on port $PORT."
    echo "Use a different port or stop the existing service."
    echo "To stop the existing service: kill \$(lsof -t -i:$PORT)"
    exit 0
fi

# Check if virtual environment exists
if [ ! -d "python/venv" ]; then
    echo "Python environment not found. Setting up..."
    ./python/setup_local_env.sh
fi

# Start the service
echo "Starting neural service on port $PORT..."
source python/venv/bin/activate
exec python python/neural_service.py --port $PORT $POLICY_WEIGHTS $VALUE_WEIGHTS 
#!/bin/bash

# Run a complete training session using GPU-accelerated MCTS

set -e

# Configuration
ITERATIONS=10000
SELF_PLAY_GAMES=100
BATCH_SIZE=64
SERVICE_ADDR="localhost:50052"
OUTPUT_DIR="output/gpu_training"

# Ensure neural service is running
if ! pgrep -f "python neural_service.py" > /dev/null; then
    echo "Starting neural service..."
    ./start_neural_service.sh &
    sleep 3 # Wait for service to start
    STARTED_SERVICE=true
else
    STARTED_SERVICE=false
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Build tools
echo "Building training tools..."
go build -o bin/gpu_training cmd/training/gpu_training.go

# Start training
echo "==================================================="
echo "Starting GPU-accelerated training with:"
echo " - Search iterations: $ITERATIONS"
echo " - Self-play games: $SELF_PLAY_GAMES"
echo " - Batch size: $BATCH_SIZE"
echo "==================================================="

# Run training with GPU acceleration
./bin/gpu_training \
    --iterations $ITERATIONS \
    --games $SELF_PLAY_GAMES \
    --batch-size $BATCH_SIZE \
    --service $SERVICE_ADDR \
    --output $OUTPUT_DIR \
    --profile

# Run performance comparison 
echo "==================================================="
echo "Running performance comparison between CPU and GPU..."
echo "==================================================="
./scripts/run_gpu_mcts_benchmark.sh

# Generate report
echo "==================================================="
echo "Generating training report..."
echo "==================================================="
python python/generate_training_report.py --input $OUTPUT_DIR --output $OUTPUT_DIR/report.html

echo "==================================================="
echo "Training completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "==================================================="

# If we started the neural service, stop it
if [ "$STARTED_SERVICE" = "true" ]; then
    echo "Stopping neural service..."
    pkill -f "python neural_service.py"
fi 
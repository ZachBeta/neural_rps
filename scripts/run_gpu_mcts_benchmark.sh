#!/bin/bash

# Run GPU MCTS benchmarks with various configurations

set -e

# Ensure neural service is running
if ! pgrep -f "python neural_service.py" > /dev/null; then
    echo "Starting neural service..."
    ./start_neural_service.sh &
    sleep 3 # Wait for service to start
fi

# Ensure benchmark tool is built
echo "Building benchmark tool..."
go build -o bin/benchmark_gpu_mcts cmd/tools/benchmark_gpu_mcts.go

# Function to run benchmark with specified parameters
run_benchmark() {
    local cpu=$1
    local iterations=$2
    local batch_size=$3
    
    local cpu_flag=""
    if [ "$cpu" = "true" ]; then
        cpu_flag="-cpu"
        echo "=========================================="
        echo "Running CPU benchmark with $iterations iterations, batch size $batch_size"
    else
        echo "=========================================="
        echo "Running GPU benchmark with $iterations iterations, batch size $batch_size"
    fi
    
    ./bin/benchmark_gpu_mcts $cpu_flag -n $iterations -batch $batch_size
    echo ""
}

# Run benchmarks with increasing iterations
echo "BENCHMARK: Increasing Iterations"
for iterations in 100 500 1000 2000 5000; do
    # GPU version
    run_benchmark false $iterations 64
    
    # CPU version
    run_benchmark true $iterations 64
done

# Run benchmarks with different batch sizes (GPU only)
echo "BENCHMARK: Different Batch Sizes (GPU only)"
for batch_size in 1 8 16 32 64 128; do
    run_benchmark false 1000 $batch_size
done

echo "All benchmarks completed!"

# If we started the neural service, stop it
if [ "$started_service" = "true" ]; then
    echo "Stopping neural service..."
    pkill -f "python neural_service.py"
fi 
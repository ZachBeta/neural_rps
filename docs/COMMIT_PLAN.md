# GPU Acceleration Commit Plan

This document outlines the files to include in our GPU acceleration implementation commit.

## New Files

1. `proto/neural_service.proto`: Protocol buffer definitions for our neural service
2. `python/neural_service.py`: Python gRPC service with TensorFlow Metal implementation
3. `python/setup_local_env.sh`: Script to set up Python environment with TensorFlow Metal
4. `python/requirements.txt`: Python dependencies
5. `python/test_client.py`: Test client for benchmarking
6. `pkg/neural/gpu/grpc_client.go`: Go client for the gRPC service
7. `GPU_ACCELERATION_IMPLEMENTATION.md`: Documentation of the implementation

## Modified Files

1. `start_neural_service.sh`: Fixed port detection and improved error reporting
2. `run_benchmark.sh`: Enhanced to properly handle port specification
3. `scripts/generate_proto.sh`: Improved protocol buffer generation
4. `go.mod`: Updated to reflect new dependencies

## Generated Files to Include

1. `proto/neural_service.pb.go`: Go protocol buffer generated code
2. `proto/neural_service_grpc.pb.go`: Go gRPC generated code
3. `proto/neural_service_pb2.py`: Python protocol buffer generated code
4. `proto/neural_service_pb2_grpc.py`: Python gRPC generated code

## Files to Delete

1. `pkg/neural/gpu/network.go`: Replaced by gRPC client
2. `pkg/neural/gpu/tensor_pool.go`: No longer needed with gRPC approach

## Commit Message

```
Implement GPU acceleration with Python gRPC service

- Added Python gRPC service using TensorFlow with Metal GPU support
- Created protocol buffer interface for neural network operations
- Implemented Go client for gRPC service
- Added benchmark and testing tools
- Documented implementation with performance results (~75x speedup for batch operations)
- Fixed issues with service startup and port detection

This implementation leverages Apple Silicon's Metal GPU through a 
Python service, providing significant performance improvements for 
batch operations (5-8x faster than CPU for large batches). Single 
operations are slower due to communication overhead, so this approach 
is optimized for MCTS and training workloads that can batch evaluations. 
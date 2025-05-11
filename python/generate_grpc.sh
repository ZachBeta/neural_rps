#!/bin/bash
# Generate Python gRPC code from protobuf definitions

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Create proto output directory if it doesn't exist
mkdir -p proto

# Generate Python code
python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    proto/neural_service.proto

echo "Python gRPC code generated successfully." 
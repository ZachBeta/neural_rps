#!/bin/bash
# Generate Go protobuf and gRPC code

set -e

# Change to the project root directory
cd "$(dirname "$0")/.."

# Create proto output directory if it doesn't exist
mkdir -p pkg/neural/proto

# Generate Go code
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/neural_service.proto

# Copy the generated files if they're not directly in pkg/neural/proto
if [ -f proto/neural_service.pb.go ] && [ ! -f pkg/neural/proto/neural_service.pb.go ]; then
    echo "Copying Go proto files to pkg/neural/proto..."
    cp proto/neural_service.pb.go pkg/neural/proto/
    cp proto/neural_service_grpc.pb.go pkg/neural/proto/
fi

# Generate Python code
python -m grpc_tools.protoc --proto_path=. \
    --python_out=. --grpc_python_out=. \
    proto/neural_service.proto

echo "Protocol buffer code generated successfully." 
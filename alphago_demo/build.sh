#!/bin/bash

echo "Building AlphaGo-Style RPS Card Game..."

# Navigate to the project root
cd "$(dirname "$0")"

# Build the RPS card game
echo "Compiling RPS card game..."
go build -o bin/rps_card ./cmd/rps_card

echo "Build complete. Run the game with: ./bin/rps_card" 
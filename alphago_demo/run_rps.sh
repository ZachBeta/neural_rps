#!/bin/bash

# Build and run the RPS card game
echo "Building RPS card game..."
go build -o rps_card cmd/rps_card/main.go
echo "Running RPS card game..."
./rps_card 
#!/bin/bash

echo "Running RPS Card Game tests..."
go test ./pkg/game/... ./pkg/neural/... ./pkg/mcts/... ./pkg/training/... -v

echo "Done!" 
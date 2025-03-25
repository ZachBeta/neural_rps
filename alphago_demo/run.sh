#!/bin/bash

# Build the game
echo "Building AlphaGo-style Tic-Tac-Toe..."
go build -o tictactoe cmd/tictactoe/main.go

# Run the game
echo "Starting the game..."
./tictactoe 
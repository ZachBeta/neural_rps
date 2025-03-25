#!/bin/bash

# Build the game
echo "Building AlphaGo-style Tic-Tac-Toe..."
go build -o tictactoe cmd/tictactoe/main.go

# Run the game and redirect output to file in parent directory
echo "Starting the game... Output will be saved to alphago_demo_output.txt"
./tictactoe | tee ../alphago_demo_output.txt 
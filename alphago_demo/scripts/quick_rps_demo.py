#!/usr/bin/env python3
"""
Quick Rock-Paper-Scissors Card Game Demonstration

This script runs a few RPS card game AI vs AI demonstrations
and saves the output to files.
"""

import os
import subprocess
import time

# Use absolute path to avoid confusion
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
NUM_GAMES = 3  # Small number of games for a quick demonstration

def run_game(game_number):
    """Run a single game in AI vs AI mode and save the output"""
    print(f"Running game {game_number}...")
    
    # Command to run the game in AI vs AI mode
    output_file = os.path.join(OUTPUT_DIR, f"quick_demo_{game_number}.txt")
    
    # Create the bin directory path
    bin_path = os.path.join(ROOT_DIR, "bin", "rps_card")
    
    # Execute the command, ensuring the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run the command
    cmd = f'(echo "2" && echo "3") | {bin_path} > {output_file}'
    subprocess.run(cmd, shell=True)
    
    print(f"Game {game_number} completed. Output saved to {output_file}")

def main():
    """Main function to run the demonstrations"""
    print(f"Running {NUM_GAMES} quick RPS card game demonstrations...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run the games
    for i in range(1, NUM_GAMES + 1):
        run_game(i)
        time.sleep(1)  # Short delay between games
    
    print(f"\nAll demonstrations completed!")
    print(f"Output saved to {OUTPUT_DIR}/quick_demo_*.txt")

if __name__ == "__main__":
    main() 
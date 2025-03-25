#!/usr/bin/env python3
"""
Analyze Rock-Paper-Scissors Card Game Demonstrations

This script runs multiple RPS card game AI vs AI demonstrations,
collects statistics, and generates a report.
"""

import os
import subprocess
import re
import time
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_DIR = "../output"
NUM_GAMES = 10  # Number of games to analyze

def run_game():
    """Run a single game in AI vs AI mode and return the output"""
    cmd = 'cd .. && (echo "2" && echo "3") | ./bin/rps_card'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8')

def parse_game_output(output):
    """Parse the game output to extract statistics"""
    # Extract the winner
    winner = "Draw"
    if "Player 1 wins!" in output:
        winner = "Player 1"
    elif "Player 2 wins!" in output:
        winner = "Player 2"
    
    # Extract number of moves
    moves = re.findall(r'Move (\d+)', output)
    num_moves = int(moves[-1]) if moves else 0
    
    # Extract card placements
    player1_cards = Counter()
    player2_cards = Counter()
    
    rock_p1 = len(re.findall(r'Player 1 plays Rock', output))
    paper_p1 = len(re.findall(r'Player 1 plays Paper', output))
    scissors_p1 = len(re.findall(r'Player 1 plays Scissors', output))
    
    rock_p2 = len(re.findall(r'Player 2 plays Rock', output))
    paper_p2 = len(re.findall(r'Player 2 plays Paper', output))
    scissors_p2 = len(re.findall(r'Player 2 plays Scissors', output))
    
    player1_cards.update({'Rock': rock_p1, 'Paper': paper_p1, 'Scissors': scissors_p1})
    player2_cards.update({'Rock': rock_p2, 'Paper': paper_p2, 'Scissors': scissors_p2})
    
    return {
        'winner': winner,
        'num_moves': num_moves,
        'player1_cards': player1_cards,
        'player2_cards': player2_cards
    }

def generate_graphs(stats):
    """Generate graphs from collected statistics"""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Winners pie chart
    winners = Counter([game['winner'] for game in stats])
    plt.figure(figsize=(10, 6))
    plt.pie([winners['Player 1'], winners['Player 2'], winners['Draw']], 
            labels=['Player 1', 'Player 2', 'Draw'],
            autopct='%1.1f%%')
    plt.title('Game Outcomes')
    plt.savefig(f"{OUTPUT_DIR}/game_outcomes.png")
    plt.close()
    
    # Card type distribution
    p1_cards = Counter()
    p2_cards = Counter()
    
    for game in stats:
        p1_cards.update(game['player1_cards'])
        p2_cards.update(game['player2_cards'])
    
    labels = ['Rock', 'Paper', 'Scissors']
    p1_values = [p1_cards[card] for card in labels]
    p2_values = [p2_cards[card] for card in labels]
    
    x = range(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], p1_values, width, label='Player 1')
    plt.bar([i + width/2 for i in x], p2_values, width, label='Player 2')
    plt.xlabel('Card Type')
    plt.ylabel('Frequency')
    plt.title('Card Type Distribution')
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/card_distribution.png")
    plt.close()
    
    # Number of moves histogram
    moves = [game['num_moves'] for game in stats]
    plt.figure(figsize=(10, 6))
    plt.hist(moves, bins=range(min(moves), max(moves) + 2), alpha=0.7)
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.title('Game Length Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/game_length.png")
    plt.close()

def generate_report(stats):
    """Generate a text report of the statistics"""
    winners = Counter([game['winner'] for game in stats])
    avg_moves = sum(game['num_moves'] for game in stats) / len(stats)
    
    p1_cards = Counter()
    p2_cards = Counter()
    
    for game in stats:
        p1_cards.update(game['player1_cards'])
        p2_cards.update(game['player2_cards'])
    
    report = f"""
RPS Card Game Analysis Report
============================
Number of games analyzed: {len(stats)}

Game Outcomes:
- Player 1 wins: {winners['Player 1']} ({winners['Player 1']/len(stats)*100:.1f}%)
- Player 2 wins: {winners['Player 2']} ({winners['Player 2']/len(stats)*100:.1f}%)
- Draws: {winners['Draw']} ({winners['Draw']/len(stats)*100:.1f}%)

Average game length: {avg_moves:.2f} moves

Card Usage:
Player 1:
- Rock: {p1_cards['Rock']}
- Paper: {p1_cards['Paper']}
- Scissors: {p1_cards['Scissors']}

Player 2:
- Rock: {p2_cards['Rock']}
- Paper: {p2_cards['Paper']}
- Scissors: {p2_cards['Scissors']}
"""
    
    # Write report to file
    with open(f"{OUTPUT_DIR}/analysis_report.txt", 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main function to run the analysis"""
    print(f"Running {NUM_GAMES} RPS card game demonstrations...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    game_stats = []
    for i in range(NUM_GAMES):
        print(f"Game {i+1}/{NUM_GAMES}...")
        output = run_game()
        
        # Save the raw output
        with open(f"{OUTPUT_DIR}/game_{i+1}_output.txt", 'w') as f:
            f.write(output)
        
        stats = parse_game_output(output)
        game_stats.append(stats)
        
        # Small delay to prevent system overload
        time.sleep(1)
    
    print("Generating statistics and graphs...")
    generate_graphs(game_stats)
    report = generate_report(game_stats)
    
    print("\nAnalysis complete!")
    print(report)

if __name__ == "__main__":
    main() 
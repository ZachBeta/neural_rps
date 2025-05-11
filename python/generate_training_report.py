#!/usr/bin/env python3

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_results_file(file_path):
    """Parse the training results file."""
    results = {}
    game_results = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract configuration
    config_match = re.search(r'Configuration:\n(.*?)\n\n', content, re.DOTALL)
    if config_match:
        config_section = config_match.group(1)
        results['iterations'] = int(re.search(r'- MCTS Iterations: (\d+)', config_section).group(1))
        results['games'] = int(re.search(r'- Self-play Games: (\d+)', config_section).group(1))
        results['batch_size'] = int(re.search(r'- Batch Size: (\d+)', config_section).group(1))
        results['service'] = re.search(r'- Service: (.+)', config_section).group(1)
    
    # Extract summary statistics
    stats_match = re.search(r'Summary Statistics:\n(.*?)\n\n', content, re.DOTALL)
    if stats_match:
        stats_section = stats_match.group(1)
        results['total_duration'] = re.search(r'- Total Duration: (.+)', stats_section).group(1)
        results['avg_game_duration'] = re.search(r'- Avg. Game Duration: (.+)', stats_section).group(1)
        results['total_nodes'] = int(re.search(r'- Total Nodes: (\d+)', stats_section).group(1))
        results['total_moves'] = int(re.search(r'- Total Moves: (\d+)', stats_section).group(1))
        results['nodes_per_second'] = float(re.search(r'- Nodes/Second: ([\d.]+)', stats_section).group(1))
        results['moves_per_second'] = float(re.search(r'- Moves/Second: ([\d.]+)', stats_section).group(1))
    
    # Extract game results
    game_section = re.search(r'Game Results:\n(.*)', content, re.DOTALL)
    if game_section:
        game_lines = game_section.group(1).strip().split('\n')
        for line in game_lines:
            match = re.search(r'Game (\d+): Winner=(.+), Moves=(\d+)', line)
            if match:
                game_number = int(match.group(1))
                winner = match.group(2)
                moves = int(match.group(3))
                game_results.append({
                    'game': game_number,
                    'winner': winner,
                    'moves': moves
                })
    
    results['games_data'] = game_results
    return results

def parse_log_file(file_path):
    """Parse the training log file to extract per-game metrics."""
    game_metrics = []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'Game (\d+) completed in (.+) \((\d+) moves, (.+)/move\)', line)
            if match:
                game_number = int(match.group(1))
                duration = match.group(2)
                moves = int(match.group(3))
                time_per_move = match.group(4)
                
                # Convert durations to seconds for plotting
                duration_seconds = 0
                if 'ms' in duration:
                    duration_seconds = float(duration.replace('ms', '')) / 1000
                elif 's' in duration:
                    duration_seconds = float(duration.replace('s', ''))
                elif 'm' in duration:
                    parts = duration.split('m')
                    duration_seconds = float(parts[0]) * 60
                    if len(parts) > 1 and 's' in parts[1]:
                        duration_seconds += float(parts[1].replace('s', ''))
                
                game_metrics.append({
                    'game': game_number,
                    'duration_seconds': duration_seconds,
                    'moves': moves,
                    'seconds_per_move': duration_seconds / moves
                })
    
    return game_metrics

def generate_plots(results, game_metrics, output_dir):
    """Generate plots from the results data."""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Game duration plot
    plt.figure(figsize=(10, 6))
    games = [gm['game'] for gm in game_metrics]
    durations = [gm['duration_seconds'] for gm in game_metrics]
    plt.plot(games, durations, marker='o', linestyle='-', color='blue')
    plt.title('Game Duration by Game Number')
    plt.xlabel('Game Number')
    plt.ylabel('Duration (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'game_duration.png'))
    plt.close()
    
    # Moves per game plot
    plt.figure(figsize=(10, 6))
    games = [gm['game'] for gm in game_metrics]
    moves = [gm['moves'] for gm in game_metrics]
    plt.plot(games, moves, marker='o', linestyle='-', color='green')
    plt.title('Moves per Game')
    plt.xlabel('Game Number')
    plt.ylabel('Number of Moves')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'moves_per_game.png'))
    plt.close()
    
    # Time per move plot
    plt.figure(figsize=(10, 6))
    games = [gm['game'] for gm in game_metrics]
    time_per_move = [gm['seconds_per_move'] for gm in game_metrics]
    plt.plot(games, time_per_move, marker='o', linestyle='-', color='red')
    plt.title('Time per Move by Game Number')
    plt.xlabel('Game Number')
    plt.ylabel('Seconds per Move')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'time_per_move.png'))
    plt.close()
    
    # Winner distribution pie chart
    plt.figure(figsize=(8, 8))
    winner_counts = {}
    for game in results['games_data']:
        winner = game['winner']
        if winner not in winner_counts:
            winner_counts[winner] = 0
        winner_counts[winner] += 1
    
    labels = list(winner_counts.keys())
    sizes = list(winner_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red', 'green', 'purple'])
    plt.axis('equal')
    plt.title('Game Winner Distribution')
    plt.savefig(os.path.join(plots_dir, 'winner_distribution.png'))
    plt.close()
    
    return plots_dir

def generate_html_report(results, game_metrics, plots_dir, output_file):
    """Generate an HTML report with the results and plots."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU-Accelerated MCTS Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .stats-container {{ display: flex; flex-wrap: wrap; }}
            .stat-box {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 10px; 
                min-width: 200px;
                background-color: #f9f9f9;
            }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .plot-container {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>GPU-Accelerated MCTS Training Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Configuration</h2>
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-value">{results['iterations']}</div>
                    <div class="stat-label">MCTS Iterations</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['games']}</div>
                    <div class="stat-label">Self-play Games</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['batch_size']}</div>
                    <div class="stat-label">Batch Size</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['service']}</div>
                    <div class="stat-label">Service Address</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-value">{results['total_duration']}</div>
                    <div class="stat-label">Total Duration</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['avg_game_duration']}</div>
                    <div class="stat-label">Avg. Game Duration</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['total_nodes']}</div>
                    <div class="stat-label">Total Nodes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['total_moves']}</div>
                    <div class="stat-label">Total Moves</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['nodes_per_second']:.2f}</div>
                    <div class="stat-label">Nodes/Second</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['moves_per_second']:.2f}</div>
                    <div class="stat-label">Moves/Second</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Graphs</h2>
            
            <div class="plot-container">
                <h3>Game Duration</h3>
                <img src="plots/game_duration.png" alt="Game Duration" width="800">
            </div>
            
            <div class="plot-container">
                <h3>Moves per Game</h3>
                <img src="plots/moves_per_game.png" alt="Moves per Game" width="800">
            </div>
            
            <div class="plot-container">
                <h3>Time per Move</h3>
                <img src="plots/time_per_move.png" alt="Time per Move" width="800">
            </div>
            
            <div class="plot-container">
                <h3>Winner Distribution</h3>
                <img src="plots/winner_distribution.png" alt="Winner Distribution" width="600">
            </div>
        </div>
        
        <div class="section">
            <h2>Game Results</h2>
            <table>
                <tr>
                    <th>Game #</th>
                    <th>Winner</th>
                    <th>Moves</th>
                    <th>Duration (s)</th>
                    <th>Time per Move (s)</th>
                </tr>
    """
    
    # Add rows for each game
    for i, game in enumerate(results['games_data']):
        if i < len(game_metrics):
            metrics = game_metrics[i]
            html_content += f"""
                <tr>
                    <td>{game['game']}</td>
                    <td>{game['winner']}</td>
                    <td>{game['moves']}</td>
                    <td>{metrics['duration_seconds']:.2f}</td>
                    <td>{metrics['seconds_per_move']:.4f}</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate training report')
    parser.add_argument('--input', required=True, help='Input directory with training results')
    parser.add_argument('--output', required=True, help='Output HTML file path')
    args = parser.parse_args()
    
    results_file = os.path.join(args.input, 'results.txt')
    log_file = os.path.join(args.input, 'training.log')
    
    # Parse results and logs
    results = parse_results_file(results_file)
    game_metrics = parse_log_file(log_file)
    
    # Generate plots
    plots_dir = generate_plots(results, game_metrics, os.path.dirname(args.output))
    
    # Generate HTML report
    generate_html_report(results, game_metrics, plots_dir, args.output)
    
    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main() 
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/agents"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

// Agent defines the interface for all game-playing agents
type Agent interface {
	GetMove(state *game.RPSGame) (game.RPSMove, error)
	Name() string
}

// ModelFile represents a pair of policy and value network files
type ModelFile struct {
	Identifier string
	PolicyPath string
	ValuePath  string
}

func main() {
	// Parse command line flags
	gamesPerPair := flag.Int("games", 30, "Number of games to play per agent pair")
	outputFile := flag.String("output", "output/tournament_with_minimax_results.csv", "Output file for results")
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	maxNetworks := flag.Int("max-networks", 3, "Maximum number of neural networks of each type to include")
	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create tournament manager
	tm := NewTournamentManager(*verbose)

	// Add random agent as baseline
	tm.AddAgent(NewRandomAgent("Random"))

	// Add minimax agents with different depths
	minimaxAgent3 := agents.NewMinimaxAgent("Minimax-3", 3, 1*time.Second, true)
	minimaxAgent3.SetVerbose(*verbose)
	tm.AddAgent(minimaxAgent3)

	minimaxAgent5 := agents.NewMinimaxAgent("Minimax-5", 5, 3*time.Second, true)
	minimaxAgent5.SetVerbose(*verbose)
	tm.AddAgent(minimaxAgent5)

	// Find available models for neural networks
	fmt.Println("Looking for model files in output directory...")

	// Add NEAT models (limit to the specified max)
	neatFiles := findModelFiles("neat")
	if len(neatFiles) > *maxNetworks {
		fmt.Printf("Found %d NEAT models, limiting to %d\n", len(neatFiles), *maxNetworks)
		neatFiles = neatFiles[:*maxNetworks]
	}

	for _, model := range neatFiles {
		name := fmt.Sprintf("NEAT-%s", model.Identifier)
		tm.AddAgent(NewNeuralAgent(name, model.PolicyPath, model.ValuePath))
		fmt.Printf("Added %s agent\n", name)
	}

	// Add AlphaGo models (limit to the specified max)
	alphaGoFiles := findModelFiles("rps_h")
	if len(alphaGoFiles) > *maxNetworks {
		fmt.Printf("Found %d AlphaGo models, limiting to %d\n", len(alphaGoFiles), *maxNetworks)
		alphaGoFiles = alphaGoFiles[:*maxNetworks]
	}

	for _, model := range alphaGoFiles {
		name := fmt.Sprintf("AlphaGo-%s", model.Identifier)
		tm.AddAgent(NewNeuralAgent(name, model.PolicyPath, model.ValuePath))
		fmt.Printf("Added %s agent\n", name)
	}

	if len(tm.Agents) < 2 {
		fmt.Println("Not enough agents found. Need at least 2 agents to run a tournament.")
		return
	}

	fmt.Printf("Starting tournament with %d agents...\n\n", len(tm.Agents))

	// Run tournament with no ELO cutoff to include all agents
	tm.RunTournament(*gamesPerPair, 0)

	// Print final rankings
	fmt.Println("\n=== Final ELO Rankings ===")
	tm.PrintRankings()

	// Save results to file
	err := tm.SaveResults(*outputFile)
	if err != nil {
		fmt.Printf("Error saving results: %v\n", err)
	} else {
		fmt.Printf("\nResults saved to %s\n", *outputFile)
	}
}

// findModelFiles searches for pairs of policy and value network files
func findModelFiles(prefix string) []ModelFile {
	// Search in both main output and extended_training directories
	directories := []string{"output", "output/extended_training"}
	var models []ModelFile

	for _, dir := range directories {
		entries, err := os.ReadDir(dir)
		if err != nil {
			fmt.Printf("Error reading directory %s: %v\n", dir, err)
			continue // Skip this directory but try others
		}

		// Map to group policy and value files by identifier
		fileMap := make(map[string]ModelFile)

		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}

			name := entry.Name()
			if strings.Contains(name, prefix) && strings.HasSuffix(name, ".model") {
				// Extract identifier and determine if it's policy or value
				isPolicyFile := strings.Contains(name, "policy")
				isValueFile := strings.Contains(name, "value")

				if !isPolicyFile && !isValueFile {
					continue
				}

				// Extract identifier (content between prefix and policy/value)
				var identifier string
				if isPolicyFile {
					identifier = strings.TrimSuffix(name, "_policy.model")
				} else {
					identifier = strings.TrimSuffix(name, "_value.model")
				}

				// Remove prefix from identifier
				if strings.HasPrefix(identifier, prefix) {
					identifier = identifier[len(prefix):]
				}

				// Ensure we have an entry for this identifier
				if _, exists := fileMap[identifier]; !exists {
					fileMap[identifier] = ModelFile{
						Identifier: identifier,
					}
				}

				// Update the appropriate path
				model := fileMap[identifier]
				if isPolicyFile {
					model.PolicyPath = fmt.Sprintf("%s/%s", dir, name)
				} else if isValueFile {
					model.ValuePath = fmt.Sprintf("%s/%s", dir, name)
				}
				fileMap[identifier] = model
			}
		}

		// Add only complete model pairs (both policy and value)
		for _, model := range fileMap {
			if model.PolicyPath != "" && model.ValuePath != "" {
				models = append(models, model)
			}
		}
	}

	// Sort models by identifier for consistency
	sort.Slice(models, func(i, j int) bool {
		return models[i].Identifier < models[j].Identifier
	})

	return models
}

// NewNeuralAgent creates an agent from neural network model files
func NewNeuralAgent(name, policyPath, valuePath string) Agent {
	policyNet := neural.NewRPSPolicyNetwork(64) // Default size
	valueNet := neural.NewRPSValueNetwork(64)   // Default size

	err := policyNet.LoadFromFile(policyPath)
	if err != nil {
		fmt.Printf("Warning: Failed to load policy network %s: %v\n", policyPath, err)
		return NewRandomAgent(fmt.Sprintf("%s-Fallback", name))
	}

	err = valueNet.LoadFromFile(valuePath)
	if err != nil {
		fmt.Printf("Warning: Failed to load value network %s: %v\n", valuePath, err)
		return NewRandomAgent(fmt.Sprintf("%s-Fallback", name))
	}

	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = 200 // Use consistent simulation count for fair comparison
	mctsEngine := mcts.NewRPSMCTS(policyNet, valueNet, mctsParams)

	return &MCTSAgent{
		name:       name,
		mctsEngine: mctsEngine,
	}
}

// NewMinimaxAgent creates a minimax agent with specified depth
func NewMinimaxAgent(name string, depth int, timeLimit time.Duration, useCache bool) Agent {
	return agents.NewMinimaxAgent(name, depth, timeLimit, useCache)
}

// NewRandomAgent creates an agent that makes random moves
func NewRandomAgent(name string) Agent {
	return &RandomAgent{name: name}
}

// MCTSAgent uses MCTS for move selection
type MCTSAgent struct {
	name       string
	mctsEngine *mcts.RPSMCTS
}

func (a *MCTSAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	a.mctsEngine.SetRootState(state)
	bestNode := a.mctsEngine.Search()

	if bestNode == nil || bestNode.Move == nil {
		validMoves := state.GetValidMoves()
		if len(validMoves) == 0 {
			return game.RPSMove{}, fmt.Errorf("no valid moves")
		}
		return validMoves[rand.Intn(len(validMoves))], nil
	}

	return *bestNode.Move, nil
}

func (a *MCTSAgent) Name() string {
	return a.name
}

// RandomAgent makes random valid moves
type RandomAgent struct {
	name string
}

func (a *RandomAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	validMoves := state.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}
	return validMoves[rand.Intn(len(validMoves))], nil
}

func (a *RandomAgent) Name() string {
	return a.name
}

// Copy TournamentManager and related code from existing implementation
// Note: This is a simplified version; the complete implementation would include all methods from cmd/elo_tournament/main.go

// TournamentManager handles matches between agents and ELO calculations
type TournamentManager struct {
	Agents      []Agent
	EloRatings  map[string]float64
	GameResults map[string]map[string]*GameRecord
	VerboseMode bool
}

// GameRecord tracks game results between two agents
type GameRecord struct {
	Wins   int
	Losses int
	Draws  int
}

// NewTournamentManager creates a new tournament manager
func NewTournamentManager(verbose bool) *TournamentManager {
	return &TournamentManager{
		Agents:      make([]Agent, 0),
		EloRatings:  make(map[string]float64),
		GameResults: make(map[string]map[string]*GameRecord),
		VerboseMode: verbose,
	}
}

// AddAgent adds an agent to the tournament
func (tm *TournamentManager) AddAgent(agent Agent) {
	tm.Agents = append(tm.Agents, agent)
	tm.EloRatings[agent.Name()] = 1500.0 // Default ELO rating
	tm.GameResults[agent.Name()] = make(map[string]*GameRecord)

	// Initialize game records for this agent
	for _, otherAgent := range tm.Agents {
		if otherAgent.Name() != agent.Name() {
			tm.GameResults[agent.Name()][otherAgent.Name()] = &GameRecord{}
			if _, exists := tm.GameResults[otherAgent.Name()][agent.Name()]; !exists {
				tm.GameResults[otherAgent.Name()][agent.Name()] = &GameRecord{}
			}
		}
	}
}

// UpdateElo updates ELO ratings based on game result
func (tm *TournamentManager) UpdateElo(winner, loser string) {
	const eloK = 32.0 // K-factor for ELO calculation

	ratingWinner := tm.EloRatings[winner]
	ratingLoser := tm.EloRatings[loser]

	// Calculate expected scores
	expectedWinner := 1.0 / (1.0 + math.Pow(10, (ratingLoser-ratingWinner)/400.0))
	expectedLoser := 1.0 / (1.0 + math.Pow(10, (ratingWinner-ratingLoser)/400.0))

	// Update ratings
	tm.EloRatings[winner] = ratingWinner + eloK*(1.0-expectedWinner)
	tm.EloRatings[loser] = ratingLoser + eloK*(0.0-expectedLoser)
}

// UpdateEloForDraw updates ELO ratings for a draw
func (tm *TournamentManager) UpdateEloForDraw(agent1, agent2 string) {
	const eloK = 32.0 // K-factor for ELO calculation

	rating1 := tm.EloRatings[agent1]
	rating2 := tm.EloRatings[agent2]

	// Calculate expected scores
	expected1 := 1.0 / (1.0 + math.Pow(10, (rating2-rating1)/400.0))
	expected2 := 1.0 / (1.0 + math.Pow(10, (rating1-rating2)/400.0))

	// Update ratings (0.5 for draw)
	tm.EloRatings[agent1] = rating1 + eloK*(0.5-expected1)
	tm.EloRatings[agent2] = rating2 + eloK*(0.5-expected2)
}

// playGame plays a single game between two agents
func (tm *TournamentManager) playGame(agent1, agent2 Agent) string {
	gameState := game.NewRPSGame(21, 5, 10) // Use standard parameters

	// Determine who goes first randomly
	firstPlayer := rand.Intn(2) == 0

	for !gameState.IsGameOver() {
		var currentAgent Agent
		if (gameState.CurrentPlayer == game.Player1 && firstPlayer) ||
			(gameState.CurrentPlayer == game.Player2 && !firstPlayer) {
			currentAgent = agent1
		} else {
			currentAgent = agent2
		}

		move, err := currentAgent.GetMove(gameState.Copy())
		if err != nil {
			if tm.VerboseMode {
				fmt.Printf("Error getting move from %s: %v\n", currentAgent.Name(), err)
			}
			// Return the other agent as winner if there's an error
			if currentAgent == agent1 {
				return agent2.Name()
			} else {
				return agent1.Name()
			}
		}

		move.Player = gameState.CurrentPlayer
		err = gameState.MakeMove(move)
		if err != nil {
			if tm.VerboseMode {
				fmt.Printf("Invalid move from %s: %v\n", currentAgent.Name(), err)
			}
			// Return the other agent as winner if there's an invalid move
			if currentAgent == agent1 {
				return agent2.Name()
			} else {
				return agent1.Name()
			}
		}
	}

	// Determine winner
	winner := gameState.GetWinner()
	if winner == game.NoPlayer {
		return "draw"
	}

	if (winner == game.Player1 && firstPlayer) || (winner == game.Player2 && !firstPlayer) {
		return agent1.Name()
	} else {
		return agent2.Name()
	}
}

// RunTournament runs a tournament between all agents
func (tm *TournamentManager) RunTournament(gamesPerPair int, eloCutoff float64) {
	fmt.Printf("Starting tournament with %d agents, %d games per pair...\n",
		len(tm.Agents), gamesPerPair)

	if eloCutoff > 0 {
		fmt.Printf("Agents with ELO below %.0f will be removed from the tournament.\n", eloCutoff)
	}

	// Active agents list (will be pruned as tournament progresses if cutoff is enabled)
	activeAgents := make([]Agent, len(tm.Agents))
	copy(activeAgents, tm.Agents)

	// Track matchups played to avoid repeats
	matchupsPlayed := make(map[string]bool)

	totalMatchups := len(activeAgents) * (len(activeAgents) - 1) / 2
	fmt.Printf("Initial matchups to play: %d\n\n", totalMatchups)

	gameCount := 0
	matchupCount := 0
	startTime := time.Now()

	// Continue until all matchups are played
	for {
		// Break if there are fewer than 2 active agents
		if len(activeAgents) < 2 {
			break
		}

		// Find next pair of agents to play
		agent1, agent2, found := tm.selectNextMatchup(activeAgents, matchupsPlayed)
		if !found {
			break // No more matchups to play
		}

		matchupKey := getMatchupKey(agent1.Name(), agent2.Name())
		matchupsPlayed[matchupKey] = true
		matchupCount++

		fmt.Printf("Match: %s (ELO: %.0f) vs %s (ELO: %.0f) - %d games\n",
			agent1.Name(), tm.EloRatings[agent1.Name()],
			agent2.Name(), tm.EloRatings[agent2.Name()],
			gamesPerPair)

		wins1, wins2, draws := 0, 0, 0

		for k := 0; k < gamesPerPair; k++ {
			result := tm.playGame(agent1, agent2)
			gameCount++

			// Update statistics and ELO ratings
			if result == agent1.Name() {
				wins1++
				tm.GameResults[agent1.Name()][agent2.Name()].Wins++
				tm.GameResults[agent2.Name()][agent1.Name()].Losses++
				tm.UpdateElo(agent1.Name(), agent2.Name())
			} else if result == agent2.Name() {
				wins2++
				tm.GameResults[agent2.Name()][agent1.Name()].Wins++
				tm.GameResults[agent1.Name()][agent2.Name()].Losses++
				tm.UpdateElo(agent2.Name(), agent1.Name())
			} else {
				draws++
				tm.GameResults[agent1.Name()][agent2.Name()].Draws++
				tm.GameResults[agent2.Name()][agent1.Name()].Draws++
				tm.UpdateEloForDraw(agent1.Name(), agent2.Name())
			}

			// Report progress every 10 games
			if gameCount%10 == 0 {
				elapsed := time.Since(startTime)
				gamesPerSec := float64(gameCount) / elapsed.Seconds()
				fmt.Printf("\rProgress: %d games (%.1f games/sec) | Matchup %d: %d-%d-%d",
					gameCount, gamesPerSec, matchupCount, wins1, wins2, draws)
			}
		}

		// Print match results
		fmt.Printf("\nResult: %s %d - %d %s (draws: %d)\n",
			agent1.Name(), wins1, wins2, agent2.Name(), draws)
		fmt.Printf("Updated ELO: %s: %.0f | %s: %.0f\n\n",
			agent1.Name(), tm.EloRatings[agent1.Name()],
			agent2.Name(), tm.EloRatings[agent2.Name()])

		// Show current leaderboard periodically
		if matchupCount%5 == 0 { // Show leaderboard every 5 matchups
			fmt.Println("\n--- Current Leaderboard ---")
			tm.PrintTopRankings(10) // Show top 10 agents
			fmt.Println()
		}

		// Prune weak agents if cutoff is enabled
		if eloCutoff > 0 {
			prunedAgents := tm.pruneWeakAgents(activeAgents, eloCutoff)
			if len(prunedAgents) < len(activeAgents) {
				activeAgents = prunedAgents
				fmt.Printf("Pruned agents below ELO %.0f. %d agents remaining.\n\n",
					eloCutoff, len(activeAgents))
			}
		}
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\nTournament completed in %s (%.1f games/sec)\n",
		elapsed, float64(gameCount)/elapsed.Seconds())
	fmt.Printf("Total games played: %d across %d matchups\n",
		gameCount, matchupCount)
}

// selectNextMatchup selects the next pair of agents to play
func (tm *TournamentManager) selectNextMatchup(agents []Agent, played map[string]bool) (agent1, agent2 Agent, found bool) {
	// Strategy: Match agents with similar ELO ratings first
	for i := 0; i < len(agents); i++ {
		for j := i + 1; j < len(agents); j++ {
			a1 := agents[i]
			a2 := agents[j]
			key := getMatchupKey(a1.Name(), a2.Name())

			if !played[key] {
				return a1, a2, true
			}
		}
	}

	return nil, nil, false
}

// getMatchupKey creates a unique key for a matchup between two agents
func getMatchupKey(name1, name2 string) string {
	// Ensure consistent ordering of names
	if name1 < name2 {
		return name1 + ":" + name2
	}
	return name2 + ":" + name1
}

// pruneWeakAgents removes agents below the ELO threshold
func (tm *TournamentManager) pruneWeakAgents(agents []Agent, threshold float64) []Agent {
	if threshold <= 0 {
		return agents // No pruning if threshold is disabled
	}

	filtered := make([]Agent, 0, len(agents))
	for _, agent := range agents {
		if tm.EloRatings[agent.Name()] >= threshold {
			filtered = append(filtered, agent)
		}
	}
	return filtered
}

// PrintTopRankings displays the top N agents by ELO rating
func (tm *TournamentManager) PrintTopRankings(n int) {
	// Sort agents by ELO rating
	type RankedAgent struct {
		Name   string
		Elo    float64
		Wins   int
		Losses int
		Draws  int
	}

	rankings := make([]RankedAgent, 0, len(tm.Agents))

	for _, agent := range tm.Agents {
		name := agent.Name()
		wins, losses, draws := 0, 0, 0

		// Calculate total wins/losses/draws
		for _, otherAgent := range tm.Agents {
			otherName := otherAgent.Name()
			if name != otherName {
				if record, exists := tm.GameResults[name][otherName]; exists {
					wins += record.Wins
					losses += record.Losses
					draws += record.Draws
				}
			}
		}

		rankings = append(rankings, RankedAgent{
			Name:   name,
			Elo:    tm.EloRatings[name],
			Wins:   wins,
			Losses: losses,
			Draws:  draws,
		})
	}

	// Sort by ELO
	sort.Slice(rankings, func(i, j int) bool {
		return rankings[i].Elo > rankings[j].Elo
	})

	// Limit to top N
	if n > 0 && n < len(rankings) {
		rankings = rankings[:n]
	}

	// Print rankings table
	fmt.Printf("%-4s %-30s %-6s %-6s %-6s %-6s %-6s\n",
		"Rank", "Agent", "ELO", "W", "L", "D", "W%")
	fmt.Println(strings.Repeat("-", 72))

	for i, agent := range rankings {
		totalGames := agent.Wins + agent.Losses + agent.Draws
		winPercentage := 0.0
		if totalGames > 0 {
			winPercentage = 100.0 * float64(agent.Wins) / float64(totalGames)
		}

		fmt.Printf("%-4d %-30s %-6.0f %-6d %-6d %-6d %-6.1f%%\n",
			i+1, agent.Name, agent.Elo, agent.Wins, agent.Losses, agent.Draws, winPercentage)
	}
}

// PrintRankings displays all agents sorted by ELO rating
func (tm *TournamentManager) PrintRankings() {
	tm.PrintTopRankings(0) // 0 means show all
}

// SaveResults saves tournament results to a file
func (tm *TournamentManager) SaveResults(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write header
	fmt.Fprintf(f, "Agent,ELO,Wins,Losses,Draws,Win%%\n")

	// Write data for each agent
	for _, agent := range tm.Agents {
		name := agent.Name()
		elo := tm.EloRatings[name]

		wins, losses, draws := 0, 0, 0
		for _, otherAgent := range tm.Agents {
			otherName := otherAgent.Name()
			if name != otherName {
				if record, exists := tm.GameResults[name][otherName]; exists {
					wins += record.Wins
					losses += record.Losses
					draws += record.Draws
				}
			}
		}

		totalGames := wins + losses + draws
		winPercentage := 0.0
		if totalGames > 0 {
			winPercentage = 100.0 * float64(wins) / float64(totalGames)
		}

		fmt.Fprintf(f, "%s,%.0f,%d,%d,%d,%.1f%%\n",
			name, elo, wins, losses, draws, winPercentage)
	}

	// Write detailed head-to-head results
	fmt.Fprintf(f, "\nHead-to-Head Results:\n")
	fmt.Fprintf(f, "Agent 1,Agent 2,Agent 1 Wins,Agent 2 Wins,Draws\n")

	for i, agent1 := range tm.Agents {
		for j, agent2 := range tm.Agents {
			if i < j {
				name1 := agent1.Name()
				name2 := agent2.Name()
				record := tm.GameResults[name1][name2]

				fmt.Fprintf(f, "%s,%s,%d,%d,%d\n",
					name1, name2, record.Wins, tm.GameResults[name2][name1].Wins, record.Draws)
			}
		}
	}

	return nil
}

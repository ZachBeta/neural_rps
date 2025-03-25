package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// Agent interface that all agents must implement
type Agent interface {
	GetMove(state *game.RPSGame) (game.RPSMove, error)
	Name() string
}

// Game structure to track a match between two agents
type GameMatch struct {
	ID           string
	Game         *game.RPSGame
	Player1Agent Agent
	Player2Agent Agent
	Winner       game.RPSPlayer
	Completed    bool
	mutex        sync.Mutex
}

// Server to manage games and connections
type GameServer struct {
	Games      map[string]*GameMatch
	Agents     map[string]Agent
	mutex      sync.Mutex
	Tournament *Tournament
	DeckSize   int
	HandSize   int
	MaxRounds  int
}

// Tournament structure to track competition results
type Tournament struct {
	Matches    map[string]*GameMatch
	Results    map[string]int // Agent name -> score
	InProgress bool
	mutex      sync.Mutex
}

// Create a new game server
func NewGameServer() *GameServer {
	return &GameServer{
		Games:      make(map[string]*GameMatch),
		Agents:     make(map[string]Agent),
		Tournament: NewTournament(),
		DeckSize:   21,
		HandSize:   5,
		MaxRounds:  10,
	}
}

// Create a new tournament
func NewTournament() *Tournament {
	return &Tournament{
		Matches:    make(map[string]*GameMatch),
		Results:    make(map[string]int),
		InProgress: false,
	}
}

// Register an agent with the server
func (s *GameServer) RegisterAgent(agent Agent) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.Agents[agent.Name()] = agent
	log.Printf("Agent registered: %s", agent.Name())
}

// Create a new game between two agents
func (s *GameServer) CreateGame(player1AgentName, player2AgentName string) (string, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	player1Agent, ok := s.Agents[player1AgentName]
	if !ok {
		return "", fmt.Errorf("agent not found: %s", player1AgentName)
	}

	player2Agent, ok := s.Agents[player2AgentName]
	if !ok {
		return "", fmt.Errorf("agent not found: %s", player2AgentName)
	}

	gameID := fmt.Sprintf("%s_vs_%s_%d", player1AgentName, player2AgentName, time.Now().UnixNano())
	gameInstance := game.NewRPSGame(s.DeckSize, s.HandSize, s.MaxRounds)

	match := &GameMatch{
		ID:           gameID,
		Game:         gameInstance,
		Player1Agent: player1Agent,
		Player2Agent: player2Agent,
		Completed:    false,
	}

	s.Games[gameID] = match
	log.Printf("Game created: %s vs %s (ID: %s)", player1AgentName, player2AgentName, gameID)
	return gameID, nil
}

// Run a single game to completion
func (s *GameServer) RunGame(gameID string) (*game.RPSGame, error) {
	match, ok := s.Games[gameID]
	if !ok {
		return nil, fmt.Errorf("game not found: %s", gameID)
	}

	match.mutex.Lock()
	defer match.mutex.Unlock()

	if match.Completed {
		return match.Game, nil
	}

	// Run the game loop
	for !match.Game.IsGameOver() {
		var currentAgent Agent
		if match.Game.CurrentPlayer == game.Player1 {
			currentAgent = match.Player1Agent
		} else {
			currentAgent = match.Player2Agent
		}

		move, err := currentAgent.GetMove(match.Game.Copy())
		if err != nil {
			return nil, fmt.Errorf("agent %s failed to make a move: %v", currentAgent.Name(), err)
		}

		move.Player = match.Game.CurrentPlayer
		err = match.Game.MakeMove(move)
		if err != nil {
			return nil, fmt.Errorf("invalid move from agent %s: %v", currentAgent.Name(), err)
		}

		log.Printf("Game %s: %s made move - Card: %d, Position: %d",
			match.ID, currentAgent.Name(), move.CardIndex, move.Position)
	}

	// Determine winner
	winner := match.Game.GetWinner()
	match.Winner = winner
	match.Completed = true

	// Log the result
	var winnerName string
	switch winner {
	case game.Player1:
		winnerName = match.Player1Agent.Name()
	case game.Player2:
		winnerName = match.Player2Agent.Name()
	default:
		winnerName = "Draw"
	}
	log.Printf("Game %s completed. Winner: %s", match.ID, winnerName)

	return match.Game, nil
}

// Start a tournament between all registered agents
func (s *GameServer) StartTournament() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.Tournament.InProgress {
		return fmt.Errorf("tournament already in progress")
	}

	// Reset the tournament
	s.Tournament = NewTournament()
	s.Tournament.InProgress = true

	// Create matches for all agent pairs (each pair plays twice, switching positions)
	agents := make([]Agent, 0, len(s.Agents))
	for _, agent := range s.Agents {
		agents = append(agents, agent)
	}

	for i := 0; i < len(agents); i++ {
		for j := 0; j < len(agents); j++ {
			if i == j {
				continue // Skip self-matches
			}

			// Create matches where each agent plays as Player1 and Player2
			gameID, err := s.CreateGame(agents[i].Name(), agents[j].Name())
			if err != nil {
				return err
			}
			s.Tournament.Matches[gameID] = s.Games[gameID]

			// Initialize scores to 0
			s.Tournament.Results[agents[i].Name()] = 0
			s.Tournament.Results[agents[j].Name()] = 0
		}
	}

	go s.runTournament()
	return nil
}

// Run all tournament matches
func (s *GameServer) runTournament() {
	s.Tournament.mutex.Lock()
	matchIDs := make([]string, 0, len(s.Tournament.Matches))
	for id := range s.Tournament.Matches {
		matchIDs = append(matchIDs, id)
	}
	s.Tournament.mutex.Unlock()

	for _, id := range matchIDs {
		_, err := s.RunGame(id)
		if err != nil {
			log.Printf("Error running tournament game %s: %v", id, err)
			continue
		}

		// Update tournament results
		s.Tournament.mutex.Lock()
		match := s.Tournament.Matches[id]
		if match.Winner == game.Player1 {
			s.Tournament.Results[match.Player1Agent.Name()] += 3 // 3 points for a win
		} else if match.Winner == game.Player2 {
			s.Tournament.Results[match.Player2Agent.Name()] += 3 // 3 points for a win
		} else {
			// Draw
			s.Tournament.Results[match.Player1Agent.Name()] += 1 // 1 point for a draw
			s.Tournament.Results[match.Player2Agent.Name()] += 1 // 1 point for a draw
		}
		s.Tournament.mutex.Unlock()
	}

	// Mark tournament as completed
	s.Tournament.mutex.Lock()
	s.Tournament.InProgress = false
	s.Tournament.mutex.Unlock()
	log.Println("Tournament completed!")
}

// Get tournament results
func (s *GameServer) GetTournamentResults() map[string]int {
	s.Tournament.mutex.Lock()
	defer s.Tournament.mutex.Unlock()

	// Make a copy of the results
	results := make(map[string]int)
	for agent, score := range s.Tournament.Results {
		results[agent] = score
	}

	return results
}

// --- HTTP Handlers ---

func (s *GameServer) handleGetAgents(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	agentNames := make([]string, 0, len(s.Agents))
	for name := range s.Agents {
		agentNames = append(agentNames, name)
	}
	s.mutex.Unlock()

	json.NewEncoder(w).Encode(agentNames)
}

func (s *GameServer) handleGetGames(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	gameIDs := make([]string, 0, len(s.Games))
	for id := range s.Games {
		gameIDs = append(gameIDs, id)
	}
	s.mutex.Unlock()

	json.NewEncoder(w).Encode(gameIDs)
}

func (s *GameServer) handleGetGameStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	gameID := vars["id"]

	s.mutex.Lock()
	match, ok := s.Games[gameID]
	s.mutex.Unlock()

	if !ok {
		http.Error(w, "Game not found", http.StatusNotFound)
		return
	}

	match.mutex.Lock()
	defer match.mutex.Unlock()

	type GameStatus struct {
		ID          string `json:"id"`
		Player1     string `json:"player1"`
		Player2     string `json:"player2"`
		CurrentTurn string `json:"current_turn,omitempty"`
		IsCompleted bool   `json:"is_completed"`
		Winner      string `json:"winner,omitempty"`
		BoardState  string `json:"board_state"`
	}

	status := GameStatus{
		ID:          match.ID,
		Player1:     match.Player1Agent.Name(),
		Player2:     match.Player2Agent.Name(),
		IsCompleted: match.Completed,
		BoardState:  match.Game.String(),
	}

	if !match.Completed {
		if match.Game.CurrentPlayer == game.Player1 {
			status.CurrentTurn = match.Player1Agent.Name()
		} else {
			status.CurrentTurn = match.Player2Agent.Name()
		}
	} else {
		switch match.Winner {
		case game.Player1:
			status.Winner = match.Player1Agent.Name()
		case game.Player2:
			status.Winner = match.Player2Agent.Name()
		default:
			status.Winner = "Draw"
		}
	}

	json.NewEncoder(w).Encode(status)
}

func (s *GameServer) handleCreateGame(w http.ResponseWriter, r *http.Request) {
	type GameRequest struct {
		Player1 string `json:"player1"`
		Player2 string `json:"player2"`
	}

	var req GameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	gameID, err := s.CreateGame(req.Player1, req.Player2)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"id": gameID})
}

func (s *GameServer) handleRunGame(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	gameID := vars["id"]

	_, err := s.RunGame(gameID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, "/api/games/"+gameID, http.StatusSeeOther)
}

func (s *GameServer) handleStartTournament(w http.ResponseWriter, r *http.Request) {
	if err := s.StartTournament(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{"status": "tournament started"})
}

func (s *GameServer) handleGetTournamentStatus(w http.ResponseWriter, r *http.Request) {
	s.Tournament.mutex.Lock()
	inProgress := s.Tournament.InProgress
	matchCount := len(s.Tournament.Matches)

	completedMatches := 0
	for _, match := range s.Tournament.Matches {
		if match.Completed {
			completedMatches++
		}
	}

	results := make(map[string]int)
	for agent, score := range s.Tournament.Results {
		results[agent] = score
	}
	s.Tournament.mutex.Unlock()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"in_progress":       inProgress,
		"total_matches":     matchCount,
		"completed_matches": completedMatches,
		"results":           results,
	})
}

func main() {
	server := NewGameServer()
	router := mux.NewRouter()

	// API routes
	router.HandleFunc("/api/agents", server.handleGetAgents).Methods("GET")
	router.HandleFunc("/api/games", server.handleGetGames).Methods("GET")
	router.HandleFunc("/api/games", server.handleCreateGame).Methods("POST")
	router.HandleFunc("/api/games/{id}", server.handleGetGameStatus).Methods("GET")
	router.HandleFunc("/api/games/{id}/run", server.handleRunGame).Methods("POST")
	router.HandleFunc("/api/tournament/start", server.handleStartTournament).Methods("POST")
	router.HandleFunc("/api/tournament/status", server.handleGetTournamentStatus).Methods("GET")

	log.Println("Starting game server on :8080")
	log.Fatal(http.ListenAndServe(":8080", router))
}

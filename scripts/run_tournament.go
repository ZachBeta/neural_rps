package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Create a game server
	server := NewGameServer()

	// Register the built-in random agent
	server.RegisterAgent(NewRandomAgent("Random"))

	// Register AlphaGo agent
	registerAlphaGoAgent(server)

	// Register Go implementation agent
	registerGoAgent(server)

	// Register C++ implementation agent
	registerCppAgent(server)

	// Start a tournament between all agents
	fmt.Println("Starting tournament between all registered agents...")
	err := server.StartTournament()
	if err != nil {
		log.Fatalf("Error starting tournament: %v", err)
	}

	// Wait for the tournament to complete
	for {
		time.Sleep(1 * time.Second)

		// Check if tournament is done
		server.Tournament.mutex.Lock()
		inProgress := server.Tournament.InProgress
		server.Tournament.mutex.Unlock()

		if !inProgress {
			break
		}
	}

	// Print tournament results
	results := server.GetTournamentResults()
	fmt.Println("\nTournament Results:")
	fmt.Println("==================")
	for agent, score := range results {
		fmt.Printf("%s: %d points\n", agent, score)
	}
}

func registerAlphaGoAgent(server *GameServer) {
	// Create policy and value networks for AlphaGo agent
	policyNetwork := neural.NewRPSPolicyNetwork(128)
	valueNetwork := neural.NewRPSValueNetwork(128)

	// Try to load pre-trained model if available
	policyErr := policyNetwork.LoadFromFile("alphago_demo/output/rps_policy.model")
	valueErr := valueNetwork.LoadFromFile("alphago_demo/output/rps_value.model")

	if policyErr != nil || valueErr != nil {
		fmt.Println("Warning: Could not load pre-trained AlphaGo models")
		fmt.Printf("Policy error: %v\n", policyErr)
		fmt.Printf("Value error: %v\n", valueErr)
	} else {
		fmt.Println("Successfully loaded pre-trained AlphaGo models")
	}

	// Register AlphaGo agent
	server.RegisterAgent(NewAlphaGoAgent("AlphaGo-MCTS", policyNetwork, valueNetwork))
}

func registerGoAgent(server *GameServer) {
	// Check if the Go agent executable exists
	goAgentPath := "golang_implementation/bin/rps_agent"
	if _, err := os.Stat(goAgentPath); os.IsNotExist(err) {
		fmt.Printf("Warning: Go agent executable not found at %s\n", goAgentPath)
		return
	}

	// If we have multiple Go agent variants, register each one
	// Register the standard Go agent
	server.RegisterAgent(NewGoExternalAgent("Go-PPO", goAgentPath, []string{"--model", "golang_implementation/models/ppo.model"}))
}

func registerCppAgent(server *GameServer) {
	// Check if the C++ agent executable exists
	cppAgentPath := "cpp_implementation/build/rps_agent"
	if _, err := os.Stat(cppAgentPath); os.IsNotExist(err) {
		fmt.Printf("Warning: C++ agent executable not found at %s\n", cppAgentPath)
		return
	}

	// Register the standard C++ agent
	server.RegisterAgent(NewCPPExternalAgent("CPP-Neural", cppAgentPath, []string{"--model", "cpp_implementation/models/neural.model"}))
}

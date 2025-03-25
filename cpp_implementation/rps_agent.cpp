#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <random>
#include <ctime>
#include "NeuralNetwork.hpp"

struct GameState {
    std::string board;
    std::string hand1;
    std::string hand2;
    int currentPlayer;
};

// Parse game state from string format
GameState parseGameState(const std::string& stateStr) {
    GameState state;
    
    // Format: Board:R.S.P...|Hand1:RPS|Hand2:RPS|Current:1
    std::map<std::string, std::string> parts;
    std::string key, value;
    size_t pos = 0;
    
    // Split by '|'
    std::string remainingStr = stateStr;
    while (!remainingStr.empty()) {
        size_t pipePos = remainingStr.find('|');
        std::string part = (pipePos != std::string::npos) ? 
                            remainingStr.substr(0, pipePos) : 
                            remainingStr;
        
        // Split by ':'
        size_t colonPos = part.find(':');
        if (colonPos != std::string::npos) {
            key = part.substr(0, colonPos);
            value = part.substr(colonPos + 1);
            parts[key] = value;
        }
        
        if (pipePos != std::string::npos) {
            remainingStr = remainingStr.substr(pipePos + 1);
        } else {
            break;
        }
    }
    
    // Populate game state
    state.board = parts["Board"];
    state.hand1 = parts["Hand1"];
    state.hand2 = parts["Hand2"];
    state.currentPlayer = std::stoi(parts["Current"]);
    
    return state;
}

// Convert game state to neural network input
std::vector<float> gameStateToInput(const GameState& state) {
    std::vector<float> input(27, 0.0f); // 9 positions * 3 features (R, P, S)
    
    // Process board state
    for (size_t i = 0; i < state.board.size() && i < 9; ++i) {
        char c = state.board[i];
        if (c == '.') {
            continue; // Empty space
        }
        
        // Determine card type and owner
        bool isPlayer1 = (c == 'R' || c == 'P' || c == 'S');
        char cardType = std::toupper(c);
        int cardIndex = -1;
        
        if (cardType == 'R') cardIndex = 0;
        else if (cardType == 'P') cardIndex = 1;
        else if (cardType == 'S') cardIndex = 2;
        
        if (cardIndex >= 0) {
            // Set position for this card type
            input[i * 3 + cardIndex] = isPlayer1 ? 1.0f : -1.0f;
        }
    }
    
    return input;
}

// Find valid moves from current state
struct Move {
    int cardIndex;
    int position;
};

std::vector<Move> getValidMoves(const GameState& state) {
    std::vector<Move> moves;
    std::string hand = (state.currentPlayer == 1) ? state.hand1 : state.hand2;
    
    // Check each position on the board
    for (int pos = 0; pos < 9; ++pos) {
        if (pos < state.board.size() && state.board[pos] == '.') {
            // Position is empty, can play any card from hand
            for (int cardIdx = 0; cardIdx < hand.size(); ++cardIdx) {
                moves.push_back({cardIdx, pos});
            }
        }
    }
    
    return moves;
}

// Choose move based on neural network output
Move chooseBestMove(NeuralNetwork& network, const GameState& state) {
    std::vector<Move> validMoves = getValidMoves(state);
    
    if (validMoves.empty()) {
        throw std::runtime_error("No valid moves available");
    }
    
    // If only one move, return it
    if (validMoves.size() == 1) {
        return validMoves[0];
    }
    
    // Convert game state to network input
    std::vector<float> input = gameStateToInput(state);
    
    // Get network prediction
    std::vector<float> output = network.forward(input);
    
    // Calculate scores for each valid move
    std::vector<std::pair<float, int>> moveScores;
    for (size_t i = 0; i < validMoves.size(); ++i) {
        const Move& move = validMoves[i];
        
        // Calculate score from network output (position is index 0-8)
        float score = output[move.position];
        moveScores.push_back({score, i});
    }
    
    // Sort moves by score (descending)
    std::sort(moveScores.begin(), moveScores.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return the highest scored move
    return validMoves[moveScores[0].second];
}

// Fallback to random move selection
Move chooseRandomMove(const GameState& state) {
    std::vector<Move> validMoves = getValidMoves(state);
    
    if (validMoves.empty()) {
        throw std::runtime_error("No valid moves available");
    }
    
    // Random number generator
    static std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<int> dist(0, validMoves.size() - 1);
    
    return validMoves[dist(rng)];
}

int main(int argc, char* argv[]) {
    try {
        std::string modelPath;
        std::string gameState;
        
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            } else if (arg == "--state" && i + 1 < argc) {
                gameState = argv[++i];
            }
        }
        
        if (gameState.empty()) {
            std::cerr << "Error: Game state not provided" << std::endl;
            return 1;
        }
        
        // Parse game state
        GameState state = parseGameState(gameState);
        
        Move bestMove;
        
        // Try to use neural network if model path is provided
        if (!modelPath.empty()) {
            try {
                // Initialize network with appropriate sizes for RPS card game
                NeuralNetwork network(27, 16, 9); // 27 inputs, 16 hidden, 9 outputs (board positions)
                
                // Load model weights if file exists
                network.loadWeights(modelPath);
                
                // Get the best move from the network
                bestMove = chooseBestMove(network, state);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Error using neural network: " << e.what() << std::endl;
                std::cerr << "Falling back to random move selection" << std::endl;
                bestMove = chooseRandomMove(state);
            }
        } else {
            // No model path, use random move
            bestMove = chooseRandomMove(state);
        }
        
        // Output the selected move in the expected format: CardIndex:Position
        std::cout << bestMove.cardIndex << ":" << bestMove.position << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 
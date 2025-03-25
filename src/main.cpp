#include "NeuralNetwork.hpp"
#include <iostream>
#include <string>
#include <vector>

// Convert Move enum to string
std::string move_to_string(Move move) {
    switch (move) {
        case Move::ROCK: return "Rock";
        case Move::PAPER: return "Paper";
        case Move::SCISSORS: return "Scissors";
        default: return "Unknown";
    }
}

// Convert string to Move enum
Move string_to_move(const std::string& move_str) {
    if (move_str == "r" || move_str == "R") return Move::ROCK;
    if (move_str == "p" || move_str == "P") return Move::PAPER;
    if (move_str == "s" || move_str == "S") return Move::SCISSORS;
    throw std::invalid_argument("Invalid move");
}

// Create one-hot encoded vector for a move
Eigen::VectorXd one_hot_encode(Move move) {
    Eigen::VectorXd encoded = Eigen::VectorXd::Zero(3);
    encoded(static_cast<int>(move)) = 1.0;
    return encoded;
}

int main() {
    // Initialize neural network
    NeuralNetwork nn;
    
    // Game state: [player_last_move (3), ai_last_move (3)]
    Eigen::VectorXd game_state = Eigen::VectorXd::Zero(6);
    Move ai_last_move = Move::ROCK;  // Default start
    
    std::cout << "Welcome to Neural Rock Paper Scissors!\n";
    std::cout << "Enter your move (R/P/S) or Q to quit\n";
    
    std::string input;
    std::vector<Eigen::VectorXd> training_inputs;
    std::vector<Eigen::VectorXd> training_targets;
    
    while (true) {
        std::cout << "\nYour move (R/P/S/Q): ";
        std::getline(std::cin, input);
        
        if (input == "q" || input == "Q") break;
        
        try {
            Move player_move = string_to_move(input);
            
            // Store the current game state for training
            training_inputs.push_back(game_state);
            
            // Determine winning move against player's move
            Move winning_move;
            switch (player_move) {
                case Move::ROCK: winning_move = Move::PAPER; break;
                case Move::PAPER: winning_move = Move::SCISSORS; break;
                case Move::SCISSORS: winning_move = Move::ROCK; break;
            }
            
            // Add the winning move as the target
            training_targets.push_back(one_hot_encode(winning_move));
            
            // Train the network on the latest move
            nn.train(training_inputs, training_targets, 0.01, 10);
            
            // Update game state for next round
            game_state.segment(0, 3) = one_hot_encode(player_move);
            game_state.segment(3, 3) = one_hot_encode(ai_last_move);
            
            // Get AI's move
            Move ai_move = nn.predict(game_state);
            ai_last_move = ai_move;
            
            std::cout << "AI plays: " << move_to_string(ai_move) << "\n";
            
            // Determine winner
            if (ai_move == player_move) {
                std::cout << "It's a tie!\n";
            } else if ((ai_move == Move::ROCK && player_move == Move::SCISSORS) ||
                      (ai_move == Move::PAPER && player_move == Move::ROCK) ||
                      (ai_move == Move::SCISSORS && player_move == Move::PAPER)) {
                std::cout << "AI wins!\n";
            } else {
                std::cout << "You win!\n";
            }
            
        } catch (const std::invalid_argument& e) {
            std::cout << "Invalid move! Please enter R, P, or S.\n";
        }
    }
    
    std::cout << "Thanks for playing!\n";
    return 0;
} 
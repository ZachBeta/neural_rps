#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>

enum class Move {
    ROCK = 0,
    PAPER = 1,
    SCISSORS = 2
};

class NeuralNetwork {
public:
    NeuralNetwork(int input_size = 6, int hidden_size = 12, int output_size = 3);
    
    // Forward pass
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    
    // Training methods
    void train(const std::vector<Eigen::VectorXd>& inputs,
              const std::vector<Eigen::VectorXd>& targets,
              double learning_rate = 0.01,
              int epochs = 1000);
    
    // Make a move based on game history
    Move predict(const Eigen::VectorXd& game_state);
    
    // Save and load model weights
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);

private:
    // Network architecture
    int input_size_;
    int hidden_size_;
    int output_size_;
    
    // Network parameters
    Eigen::MatrixXd weights1_;  // Input -> Hidden
    Eigen::VectorXd bias1_;
    Eigen::MatrixXd weights2_;  // Hidden -> Output
    Eigen::VectorXd bias2_;
    
    // Activation functions
    Eigen::VectorXd relu(const Eigen::VectorXd& x);
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);
    Eigen::VectorXd softmax(const Eigen::VectorXd& x);
    
    // Random number generator
    std::mt19937 rng_;
};

#endif // NEURAL_NETWORK_HPP 
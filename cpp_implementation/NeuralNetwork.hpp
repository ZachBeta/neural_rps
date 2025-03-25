#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <random>

enum class Move {
    ROCK = 0,
    PAPER = 1,
    SCISSORS = 2
};

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    Move predict(const Eigen::VectorXd& game_state);
    
    void train(const std::vector<Eigen::VectorXd>& inputs,
               const std::vector<Eigen::VectorXd>& targets,
               double learning_rate = 0.01,
               int epochs = 100);
    
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
    
private:
    int input_size_;
    int hidden_size_;
    int output_size_;
    
    Eigen::MatrixXd weights1_;
    Eigen::VectorXd bias1_;
    Eigen::MatrixXd weights2_;
    Eigen::VectorXd bias2_;
    
    std::mt19937 rng_;
    
    Eigen::VectorXd relu(const Eigen::VectorXd& x);
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);
    Eigen::VectorXd softmax(const Eigen::VectorXd& x);
}; 
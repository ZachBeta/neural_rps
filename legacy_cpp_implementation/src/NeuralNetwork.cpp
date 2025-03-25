#include "NeuralNetwork.hpp"
#include <fstream>
#include <iostream>
#include <random>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      rng_(std::random_device{}()) {
    
    // Initialize weights with Xavier initialization
    double w1_bound = sqrt(6.0 / (input_size_ + hidden_size_));
    double w2_bound = sqrt(6.0 / (hidden_size_ + output_size_));
    
    weights1_ = Eigen::MatrixXd::Random(hidden_size_, input_size_) * w1_bound;
    bias1_ = Eigen::VectorXd::Zero(hidden_size_);
    weights2_ = Eigen::MatrixXd::Random(output_size_, hidden_size_) * w2_bound;
    bias2_ = Eigen::VectorXd::Zero(output_size_);
}

Eigen::VectorXd NeuralNetwork::relu(const Eigen::VectorXd& x) {
    return x.array().max(0.0);
}

Eigen::VectorXd NeuralNetwork::relu_derivative(const Eigen::VectorXd& x) {
    return (x.array() > 0.0).cast<double>();
}

Eigen::VectorXd NeuralNetwork::softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = x.array().exp();
    return exp_x.array() / exp_x.sum();
}

Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& input) {
    // Hidden layer
    Eigen::VectorXd hidden = weights1_ * input + bias1_;
    hidden = relu(hidden);
    
    // Output layer
    Eigen::VectorXd output = weights2_ * hidden + bias2_;
    return softmax(output);
}

Move NeuralNetwork::predict(const Eigen::VectorXd& game_state) {
    Eigen::VectorXd output = forward(game_state);
    int move_idx;
    output.maxCoeff(&move_idx);
    return static_cast<Move>(move_idx);
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs,
                         const std::vector<Eigen::VectorXd>& targets,
                         double learning_rate,
                         int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            Eigen::VectorXd hidden = weights1_ * inputs[i] + bias1_;
            Eigen::VectorXd hidden_activated = relu(hidden);
            Eigen::VectorXd output = weights2_ * hidden_activated + bias2_;
            Eigen::VectorXd predictions = softmax(output);
            
            // Compute loss
            total_loss -= (targets[i].array() * predictions.array().log()).sum();
            
            // Backward pass
            Eigen::VectorXd output_error = predictions - targets[i];
            Eigen::VectorXd hidden_error = weights2_.transpose() * output_error;
            hidden_error = hidden_error.array() * relu_derivative(hidden).array();
            
            // Update weights and biases
            weights2_ -= learning_rate * output_error * hidden_activated.transpose();
            bias2_ -= learning_rate * output_error;
            weights1_ -= learning_rate * hidden_error * inputs[i].transpose();
            bias1_ -= learning_rate * hidden_error;
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

void NeuralNetwork::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Save dimensions
    file.write(reinterpret_cast<const char*>(&input_size_), sizeof(input_size_));
    file.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    file.write(reinterpret_cast<const char*>(&output_size_), sizeof(output_size_));
    
    // Save weights and biases
    file.write(reinterpret_cast<const char*>(weights1_.data()), weights1_.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(bias1_.data()), bias1_.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(weights2_.data()), weights2_.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(bias2_.data()), bias2_.size() * sizeof(double));
}

void NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    // Load and verify dimensions
    int input_size, hidden_size, output_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
    
    if (input_size != input_size_ || hidden_size != hidden_size_ || output_size != output_size_) {
        throw std::runtime_error("Model architecture mismatch in file: " + filename);
    }
    
    // Load weights and biases
    file.read(reinterpret_cast<char*>(weights1_.data()), weights1_.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(bias1_.data()), bias1_.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(weights2_.data()), weights2_.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(bias2_.data()), bias2_.size() * sizeof(double));
} 
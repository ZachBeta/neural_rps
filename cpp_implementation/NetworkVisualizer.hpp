#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <limits>
#include <algorithm>

class NetworkVisualizer {
public:
    static void initFileOutput(const std::string& filename = "training_output.txt");
    static void closeFileOutput();
    static std::ofstream& getOutputFile();
    
    static void visualizeArchitecture(const std::vector<int>& layer_sizes, 
                                    const std::vector<std::string>& layer_names,
                                    bool to_file = false);
    
    static void visualizeWeights(const Eigen::MatrixXd& weights,
                               const std::vector<std::string>& input_labels,
                               const std::vector<std::string>& output_labels,
                               bool to_file = false);
    
    static void visualizeActionProbs(const Eigen::VectorXd& probs,
                                   const std::vector<std::string>& action_labels,
                                   bool to_file = false);
    
    static void visualizeTrainingProgress(const std::vector<float>& rewards,
                                        int window_size = 100,
                                        bool to_file = false);

private:
    static std::ofstream output_file_;
    
    // Safely compute softmax to avoid numerical issues
    static Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
        Eigen::VectorXd result = x;
        double max_val = result.maxCoeff();
        result.array() -= max_val; // Subtract max for numerical stability
        result = result.array().exp();
        result /= result.sum();
        return result;
    }
};

// Static member initialization
std::ofstream NetworkVisualizer::output_file_;

inline void NetworkVisualizer::initFileOutput(const std::string& filename) {
    output_file_.open(filename);
    if (!output_file_.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
    }
}

inline void NetworkVisualizer::closeFileOutput() {
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

inline std::ofstream& NetworkVisualizer::getOutputFile() {
    return output_file_;
}

inline void NetworkVisualizer::visualizeArchitecture(const std::vector<int>& layer_sizes, 
                                                 const std::vector<std::string>& layer_names,
                                                 bool to_file) {
    std::ostream& out = to_file ? output_file_ : std::cout;
    
    out << "\nNetwork Architecture:\n";
    out << "-------------------\n";
    
    for (size_t i = 0; i < layer_sizes.size(); i++) {
        out << layer_names[i] << " Layer: " << layer_sizes[i] << " neurons\n";
    }
    
    out << "\nLayer Connections:\n";
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        out << layer_names[i] << " -> " << layer_names[i + 1] << ": " 
            << layer_sizes[i] * layer_sizes[i + 1] << " connections\n";
    }
    
    out << "-------------------\n\n";
}

inline void NetworkVisualizer::visualizeWeights(const Eigen::MatrixXd& weights,
                                            const std::vector<std::string>& input_labels,
                                            const std::vector<std::string>& output_labels,
                                            bool to_file) {
    std::ostream& out = to_file ? output_file_ : std::cout;
    
    out << "\nNetwork Weights:\n";
    out << "--------------\n";
    
    // Print header
    out << std::setw(10) << "";
    for (const auto& label : input_labels) {
        out << std::setw(10) << label;
    }
    out << "\n";
    
    // Print weights
    for (int i = 0; i < weights.rows(); i++) {
        out << std::setw(10) << output_labels[i];
        for (int j = 0; j < weights.cols(); j++) {
            out << std::setw(10) << std::fixed << std::setprecision(3) << weights(i, j);
        }
        out << "\n";
    }
    
    out << "--------------\n\n";
}

inline void NetworkVisualizer::visualizeActionProbs(const Eigen::VectorXd& probs,
                                                const std::vector<std::string>& action_labels,
                                                bool to_file) {
    std::ostream& out = to_file ? output_file_ : std::cout;
    
    out << "\nAction Probabilities:\n";
    out << "-------------------\n";
    
    for (int i = 0; i < probs.size(); i++) {
        out << action_labels[i] << ": " << std::fixed << std::setprecision(3) << probs(i);
        
        // Visual bar
        int bar_length = static_cast<int>(probs(i) * 50);
        out << " |";
        for (int j = 0; j < bar_length; j++) {
            out << "=";
        }
        out << "\n";
    }
    
    out << "-------------------\n\n";
}

inline void NetworkVisualizer::visualizeTrainingProgress(const std::vector<float>& rewards,
                                                     int window_size,
                                                     bool to_file) {
    std::ostream& out = to_file ? output_file_ : std::cout;
    
    out << "\nTraining Progress:\n";
    out << "----------------\n";
    
    // Calculate moving average
    std::vector<float> moving_avg;
    for (size_t i = window_size; i <= rewards.size(); i++) {
        float sum = 0.0f;
        for (size_t j = i - window_size; j < i; j++) {
            sum += rewards[j];
        }
        moving_avg.push_back(sum / window_size);
    }
    
    // Find max and min for scaling
    float max_reward = *std::max_element(moving_avg.begin(), moving_avg.end());
    float min_reward = *std::min_element(moving_avg.begin(), moving_avg.end());
    float range = max_reward - min_reward;
    
    // Print header
    out << "Episode" << std::setw(15) << "Avg Reward" << std::setw(10) << "Progress\n";
    
    // Print progress at regular intervals
    int step = std::max(1, static_cast<int>(moving_avg.size() / 10));
    for (size_t i = 0; i < moving_avg.size(); i += step) {
        int episode = (i + window_size);
        float avg_reward = moving_avg[i];
        
        out << std::setw(7) << episode << std::setw(15) << std::fixed << std::setprecision(3) << avg_reward;
        
        // Visual bar
        float normalized = (range > 0) ? (avg_reward - min_reward) / range : 0.5f;
        int bar_length = static_cast<int>(normalized * 30);
        out << std::setw(10) << "|";
        for (int j = 0; j < bar_length; j++) {
            out << "=";
        }
        out << "\n";
    }
    
    out << "----------------\n\n";
} 
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

class NetworkVisualizer {
public:
    // Visualize weight matrix with labels
    static void visualizeWeights(const Eigen::MatrixXd& weights,
                               const std::vector<std::string>& input_labels,
                               const std::vector<std::string>& output_labels) {
        std::cout << "\nWeight Matrix Visualization:\n";
        std::cout << "Input features: " << input_labels.size() << ", Output features: " << output_labels.size() << "\n\n";

        // Print column headers (input labels)
        std::cout << std::setw(12) << " ";
        for (const auto& label : input_labels) {
            std::cout << std::setw(8) << label;
        }
        std::cout << "\n";

        // Print separator
        std::cout << std::string(12 + input_labels.size() * 8, '-') << "\n";

        // Print weights with row labels
        for (int i = 0; i < weights.rows(); ++i) {
            std::cout << std::setw(10) << output_labels[i] << " |";
            for (int j = 0; j < weights.cols(); ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << weights(i, j);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Visualize action probabilities
    static void visualizeActionProbs(const Eigen::VectorXd& probs,
                                   const std::vector<std::string>& action_labels) {
        std::cout << "\nAction Probabilities:\n";
        std::cout << std::string(40, '-') << "\n";

        for (int i = 0; i < probs.size(); ++i) {
            std::cout << std::setw(10) << action_labels[i] << " |";
            
            // Create probability bar
            int bar_length = static_cast<int>(probs(i) * 30);
            std::cout << " " << std::fixed << std::setprecision(3) << probs(i) << " ";
            std::cout << std::string(bar_length, '#') << "\n";
        }
        std::cout << "\n";
    }

    // Visualize network architecture
    static void visualizeArchitecture(const std::vector<int>& layer_sizes,
                                    const std::vector<std::string>& layer_names) {
        std::cout << "\nNetwork Architecture:\n";
        std::cout << std::string(50, '=') << "\n";

        int max_layer_size = *std::max_element(layer_sizes.begin(), layer_sizes.end());
        int max_height = max_layer_size * 2 + 1;

        for (int h = 0; h < max_height; ++h) {
            for (size_t l = 0; l < layer_sizes.size(); ++l) {
                int layer_size = layer_sizes[l];
                int start = (max_height - layer_size * 2) / 2;
                int end = start + layer_size * 2;

                if (h >= start && h < end && (h - start) % 2 == 0) {
                    std::cout << " (O) ";
                } else if (h == max_height - 1) {
                    std::cout << std::setw(5) << layer_names[l];
                } else {
                    std::cout << "     ";
                }

                // Add connecting lines between layers
                if (l < layer_sizes.size() - 1) {
                    std::cout << "-";
                }
            }
            std::cout << "\n";
        }
        std::cout << std::string(50, '=') << "\n";
    }

    // Track and visualize training progress
    static void visualizeTrainingProgress(const std::vector<float>& rewards,
                                        int window_size = 100) {
        std::cout << "\nTraining Progress:\n";
        std::cout << std::string(50, '=') << "\n";

        // Calculate moving average
        std::vector<float> moving_avg;
        for (size_t i = window_size; i <= rewards.size(); ++i) {
            float sum = 0;
            for (size_t j = i - window_size; j < i; ++j) {
                sum += rewards[j];
            }
            moving_avg.push_back(sum / window_size);
        }

        // Find min and max for scaling
        float min_reward = *std::min_element(moving_avg.begin(), moving_avg.end());
        float max_reward = *std::max_element(moving_avg.begin(), moving_avg.end());
        float range = max_reward - min_reward;

        // Print progress graph
        int graph_width = 40;
        for (float avg : moving_avg) {
            int pos = static_cast<int>((avg - min_reward) / range * graph_width);
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << avg << " |";
            std::cout << std::string(pos, '#') << "\n";
        }
        std::cout << std::string(50, '=') << "\n";
    }
}; 
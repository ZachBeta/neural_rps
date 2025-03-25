#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

class NetworkVisualizer {
public:
    // Add file output stream
    static std::ofstream& getOutputFile() {
        static std::ofstream file("build/training_demo.txt");
        return file;
    }

    // Visualize weight matrix with labels
    static void visualizeWeights(const Eigen::MatrixXd& weights,
                               const std::vector<std::string>& input_labels,
                               const std::vector<std::string>& output_labels,
                               bool to_file = false) {
        std::ostream& out = to_file ? getOutputFile() : std::cout;
        
        out << "\nWeight Matrix Visualization:\n";
        out << "Input features: " << input_labels.size() << ", Output features: " << output_labels.size() << "\n\n";

        // Print column headers (input labels)
        out << std::setw(12) << " ";
        for (const auto& label : input_labels) {
            out << std::setw(8) << label;
        }
        out << "\n";

        // Print separator
        out << std::string(12 + input_labels.size() * 8, '-') << "\n";

        // Print weights with row labels
        for (int i = 0; i < weights.rows(); ++i) {
            out << std::setw(10) << output_labels[i] << " |";
            for (int j = 0; j < weights.cols(); ++j) {
                out << std::setw(8) << std::fixed << std::setprecision(3) << weights(i, j);
            }
            out << "\n";
        }
        out << "\n";
    }

    // Visualize action probabilities
    static void visualizeActionProbs(const Eigen::VectorXd& probs,
                                   const std::vector<std::string>& action_labels,
                                   bool to_file = false) {
        std::ostream& out = to_file ? getOutputFile() : std::cout;
        
        out << "\nAction Probabilities:\n";
        out << std::string(40, '-') << "\n";

        for (int i = 0; i < probs.size(); ++i) {
            out << std::setw(10) << action_labels[i] << " |";
            
            // Create probability bar
            int bar_length = static_cast<int>(probs(i) * 30);
            out << " " << std::fixed << std::setprecision(3) << probs(i) << " ";
            out << std::string(bar_length, '#') << "\n";
        }
        out << "\n";
    }

    // Visualize network architecture
    static void visualizeArchitecture(const std::vector<int>& layer_sizes,
                                    const std::vector<std::string>& layer_names,
                                    bool to_file = false) {
        std::ostream& out = to_file ? getOutputFile() : std::cout;
        
        out << "\nNetwork Architecture:\n";
        out << std::string(50, '=') << "\n";

        int max_layer_size = *std::max_element(layer_sizes.begin(), layer_sizes.end());
        int max_height = max_layer_size * 2 + 1;

        for (int h = 0; h < max_height; ++h) {
            for (size_t l = 0; l < layer_sizes.size(); ++l) {
                int layer_size = layer_sizes[l];
                int start = (max_height - layer_size * 2) / 2;
                int end = start + layer_size * 2;

                if (h >= start && h < end && (h - start) % 2 == 0) {
                    out << " (O) ";
                } else if (h == max_height - 1) {
                    out << std::setw(5) << layer_names[l];
                } else {
                    out << "     ";
                }

                // Add connecting lines between layers
                if (l < layer_sizes.size() - 1) {
                    out << "-";
                }
            }
            out << "\n";
        }
        out << std::string(50, '=') << "\n";
    }

    // Track and visualize training progress
    static void visualizeTrainingProgress(const std::vector<float>& rewards,
                                        int window_size = 100,
                                        bool to_file = false) {
        std::ostream& out = to_file ? getOutputFile() : std::cout;
        
        out << "\nTraining Progress:\n";
        out << std::string(50, '=') << "\n";

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
            out << std::setw(8) << std::fixed << std::setprecision(2) << avg << " |";
            out << std::string(pos, '#') << "\n";
        }
        out << std::string(50, '=') << "\n";
    }

    // Initialize file output
    static void initFileOutput() {
        getOutputFile().open("training_demo.txt");
    }

    // Close file output
    static void closeFileOutput() {
        getOutputFile().close();
    }
}; 
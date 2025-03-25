#include "NeuralNetwork.hpp"
#include <gtest/gtest.h>

TEST(NeuralNetworkTest, Initialization) {
    NeuralNetwork nn;
    EXPECT_NO_THROW(nn.predict(Eigen::VectorXd::Zero(6)));
}

TEST(NeuralNetworkTest, PredictionShape) {
    NeuralNetwork nn;
    Eigen::VectorXd input = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd output = nn.forward(input);
    EXPECT_EQ(output.size(), 3);
}

TEST(NeuralNetworkTest, SoftmaxOutput) {
    NeuralNetwork nn;
    Eigen::VectorXd input = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd output = nn.forward(input);
    
    // Check if probabilities sum to 1 (within floating-point precision)
    EXPECT_NEAR(output.sum(), 1.0, 1e-6);
    
    // Check if all probabilities are between 0 and 1
    for (int i = 0; i < output.size(); ++i) {
        EXPECT_GE(output(i), 0.0);
        EXPECT_LE(output(i), 1.0);
    }
}

TEST(NeuralNetworkTest, Training) {
    NeuralNetwork nn;
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;
    
    // Create a simple training example
    inputs.push_back(Eigen::VectorXd::Zero(6));
    Eigen::VectorXd target = Eigen::VectorXd::Zero(3);
    target(0) = 1.0;  // Target is ROCK
    targets.push_back(target);
    
    // Training should not throw any exceptions
    EXPECT_NO_THROW(nn.train(inputs, targets, 0.01, 10));
} 
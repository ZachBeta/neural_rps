#include "PPOAgent.hpp"
#include <gtest/gtest.h>

TEST(PPOAgentTest, Initialization) {
    PPOAgent agent(9, 3);  // 9 state dims, 3 actions
    
    // Test initial policy output
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    Eigen::VectorXd probs = agent.getPolicyProbs(state);
    
    EXPECT_EQ(probs.size(), 3);
    
    // Probabilities should sum to 1
    EXPECT_NEAR(probs.sum(), 1.0, 1e-6);
    
    // All probabilities should be between 0 and 1
    for (int i = 0; i < probs.size(); i++) {
        EXPECT_GE(probs(i), 0.0);
        EXPECT_LE(probs(i), 1.0);
    }
}

TEST(PPOAgentTest, ActionSampling) {
    PPOAgent agent(9, 3);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    
    // Test with all actions valid
    std::vector<int> valid_actions = {0, 1, 2};
    int action = agent.sampleAction(state, valid_actions);
    EXPECT_GE(action, 0);
    EXPECT_LT(action, 3);
    
    // Test with restricted valid actions
    valid_actions = {1, 2};
    action = agent.sampleAction(state, valid_actions);
    EXPECT_GE(action, 1);
    EXPECT_LT(action, 3);
    
    // Test with single valid action
    valid_actions = {1};
    action = agent.sampleAction(state, valid_actions);
    EXPECT_EQ(action, 1);
}

TEST(PPOAgentTest, ValueEstimation) {
    PPOAgent agent(9, 3);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    
    // Initial value should be finite
    float value = agent.getValue(state);
    EXPECT_FALSE(std::isnan(value));
    EXPECT_FALSE(std::isinf(value));
}

TEST(PPOAgentTest, PolicyUpdate) {
    PPOAgent agent(9, 3);
    
    // Create some fake experience
    std::vector<Eigen::VectorXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    
    // Add a single transition
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    states.push_back(state);
    actions.push_back(0);
    rewards.push_back(1.0f);
    values.push_back(0.0f);
    
    // Get initial probabilities
    Eigen::VectorXd initial_probs = agent.getPolicyProbs(state);
    
    // Update the policy
    agent.update(states, actions, rewards, values, 0.1f);
    
    // Get updated probabilities
    Eigen::VectorXd updated_probs = agent.getPolicyProbs(state);
    
    // Probabilities should change after update
    EXPECT_NE((initial_probs - updated_probs).norm(), 0.0);
    
    // Probabilities should still sum to 1
    EXPECT_NEAR(updated_probs.sum(), 1.0, 1e-6);
}

TEST(PPOAgentTest, LearningConsistency) {
    PPOAgent agent(9, 3);
    
    // Create repeated positive experience for action 0
    std::vector<Eigen::VectorXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    
    // Add multiple similar transitions
    for (int i = 0; i < 10; i++) {
        states.push_back(state);
        actions.push_back(0);
        rewards.push_back(1.0f);
        values.push_back(0.0f);
    }
    
    // Get initial probability of action 0
    float initial_prob = agent.getPolicyProbs(state)(0);
    
    // Update policy multiple times
    for (int i = 0; i < 5; i++) {
        agent.update(states, actions, rewards, values, 0.1f);
    }
    
    // Get final probability of action 0
    float final_prob = agent.getPolicyProbs(state)(0);
    
    // Probability of the rewarded action should increase
    EXPECT_GT(final_prob, initial_prob);
} 
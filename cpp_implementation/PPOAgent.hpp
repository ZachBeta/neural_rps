#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

class PPOAgent {
public:
    PPOAgent(int state_dim, int action_dim);
    
    int sampleAction(const Eigen::VectorXd& state, const std::vector<int>& valid_actions);
    float getValue(const Eigen::VectorXd& state);
    void update(const std::vector<Eigen::VectorXd>& states,
                const std::vector<int>& actions,
                const std::vector<float>& rewards,
                const std::vector<float>& values);
    Eigen::VectorXd getPolicyProbs(const Eigen::VectorXd& state);
    Eigen::MatrixXd getPolicyWeights() const;
    
private:
    int state_dim_;
    int action_dim_;
    
    // Policy network weights
    Eigen::MatrixXd policy_weights_;
    
    // Value network weights
    Eigen::MatrixXd value_weights_;
    
    // Learning rates
    float policy_lr_ = 0.01f;
    float value_lr_ = 0.01f;
    
    // PPO clipping parameter
    float clip_param_ = 0.2f;
    
    // Random number generator
    std::mt19937 rng_;
    
    Eigen::VectorXd softmax(const Eigen::VectorXd& x);
    Eigen::VectorXd computeReturns(const std::vector<float>& rewards);
    Eigen::VectorXd maskInvalidActions(const Eigen::VectorXd& probs, const std::vector<int>& valid_actions);
};

inline PPOAgent::PPOAgent(int state_dim, int action_dim)
    : state_dim_(state_dim), action_dim_(action_dim), rng_(std::random_device{}()) {
    // Initialize policy network weights
    policy_weights_ = Eigen::MatrixXd::Random(action_dim_, state_dim_) * 0.1;
    
    // Initialize value network weights
    value_weights_ = Eigen::MatrixXd::Random(1, state_dim_) * 0.1;
}

inline int PPOAgent::sampleAction(const Eigen::VectorXd& state, const std::vector<int>& valid_actions) {
    Eigen::VectorXd probs = getPolicyProbs(state);
    probs = maskInvalidActions(probs, valid_actions);
    
    // Sample from the probability distribution
    std::discrete_distribution<int> dist(probs.data(), probs.data() + probs.size());
    return dist(rng_);
}

inline float PPOAgent::getValue(const Eigen::VectorXd& state) {
    return (value_weights_ * state)(0, 0);
}

inline void PPOAgent::update(const std::vector<Eigen::VectorXd>& states,
                          const std::vector<int>& actions,
                          const std::vector<float>& rewards,
                          const std::vector<float>& values) {
    // Compute returns
    Eigen::VectorXd returns = computeReturns(rewards);
    
    // Compute advantages
    Eigen::VectorXd advantages(returns.size());
    for (size_t i = 0; i < returns.size(); i++) {
        advantages(i) = returns(i) - values[i];
    }
    
    // Update policy and value networks
    for (size_t i = 0; i < states.size(); i++) {
        const Eigen::VectorXd& state = states[i];
        int action = actions[i];
        
        // Get probabilities under current policy
        Eigen::VectorXd probs = getPolicyProbs(state);
        float old_prob = probs(action);
        
        // Update policy weights
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(action_dim_);
        grad(action) = advantages(i) / std::max(old_prob, 1e-8f);
        
        // Clip gradient
        grad(action) = std::min(std::max(grad(action), -clip_param_), clip_param_);
        
        // Apply gradient to update policy weights
        for (int a = 0; a < action_dim_; a++) {
            policy_weights_.row(a) += policy_lr_ * grad(a) * state.transpose();
        }
        
        // Update value weights
        float value_error = returns(i) - values[i];
        value_weights_ += value_lr_ * value_error * state.transpose();
    }
}

inline Eigen::VectorXd PPOAgent::getPolicyProbs(const Eigen::VectorXd& state) {
    Eigen::VectorXd logits = policy_weights_ * state;
    return softmax(logits);
}

inline Eigen::MatrixXd PPOAgent::getPolicyWeights() const {
    return policy_weights_;
}

inline Eigen::VectorXd PPOAgent::softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = x.array().exp();
    return exp_x / exp_x.sum();
}

inline Eigen::VectorXd PPOAgent::computeReturns(const std::vector<float>& rewards) {
    float gamma = 0.99f; // Discount factor
    
    Eigen::VectorXd returns(rewards.size());
    float cumulative_return = 0.0f;
    
    for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; i--) {
        cumulative_return = rewards[i] + gamma * cumulative_return;
        returns(i) = cumulative_return;
    }
    
    return returns;
}

inline Eigen::VectorXd PPOAgent::maskInvalidActions(const Eigen::VectorXd& probs, const std::vector<int>& valid_actions) {
    Eigen::VectorXd masked_probs = Eigen::VectorXd::Zero(probs.size());
    
    // Set probabilities for valid actions
    float total_prob = 0.0f;
    for (int action : valid_actions) {
        masked_probs(action) = probs(action);
        total_prob += probs(action);
    }
    
    // Normalize probabilities
    if (total_prob > 0.0f) {
        masked_probs /= total_prob;
    } else {
        // If all probabilities are zero, use uniform distribution over valid actions
        for (int action : valid_actions) {
            masked_probs(action) = 1.0f / valid_actions.size();
        }
    }
    
    return masked_probs;
} 
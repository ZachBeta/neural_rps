#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <memory>

class PPOAgent {
public:
    PPOAgent(int state_dim, int action_dim)
        : state_dim_(state_dim),
          action_dim_(action_dim),
          rng_(std::random_device{}()) {
        
        // Initialize policy network
        policy_weights_ = Eigen::MatrixXd::Random(action_dim_, state_dim_) * 0.1;
        policy_bias_ = Eigen::VectorXd::Zero(action_dim_);
        
        // Initialize value network
        value_weights_ = Eigen::MatrixXd::Random(1, state_dim_) * 0.1;
        value_bias_ = Eigen::VectorXd::Zero(1);
        
        // PPO hyperparameters
        epsilon_ = 0.2;  // Clipping parameter
        c1_ = 1.0;      // Value loss coefficient
        c2_ = 0.01;     // Entropy coefficient
    }

    // Get action probabilities from policy network
    Eigen::VectorXd getPolicyProbs(const Eigen::VectorXd& state) const {
        Eigen::VectorXd logits = policy_weights_ * state + policy_bias_;
        return softmax(logits);
    }

    // Get state value from value network
    float getValue(const Eigen::VectorXd& state) const {
        return float((value_weights_ * state + value_bias_)(0));
    }

    // Sample an action from the policy
    int sampleAction(const Eigen::VectorXd& state, const std::vector<int>& valid_actions) {
        Eigen::VectorXd probs = getPolicyProbs(state);
        
        // Mask invalid actions
        Eigen::VectorXd masked_probs = Eigen::VectorXd::Zero(action_dim_);
        float sum = 0.0f;
        for (int action : valid_actions) {
            masked_probs(action) = probs(action);
            sum += probs(action);
        }
        
        // Renormalize
        if (sum > 0) {
            masked_probs /= sum;
        } else {
            // If all probabilities were zero, use uniform distribution
            for (int action : valid_actions) {
                masked_probs(action) = 1.0f / valid_actions.size();
            }
        }
        
        // Sample from distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cumsum = 0.0f;
        
        for (int i = 0; i < action_dim_; i++) {
            cumsum += masked_probs(i);
            if (r <= cumsum) {
                return i;
            }
        }
        
        return valid_actions[0];  // Fallback
    }

    // Update networks using PPO
    void update(const std::vector<Eigen::VectorXd>& states,
               const std::vector<int>& actions,
               const std::vector<float>& rewards,
               const std::vector<float>& values,
               float learning_rate = 0.001) {
        
        // Compute advantages and returns
        std::vector<float> advantages;
        std::vector<float> returns;
        computeAdvantagesAndReturns(rewards, values, advantages, returns);
        
        // Get old action probabilities
        std::vector<float> old_probs;
        for (size_t i = 0; i < states.size(); i++) {
            Eigen::VectorXd probs = getPolicyProbs(states[i]);
            old_probs.push_back(probs(actions[i]));
        }
        
        // PPO update
        for (size_t i = 0; i < states.size(); i++) {
            // Policy gradient update
            Eigen::VectorXd probs = getPolicyProbs(states[i]);
            float prob_ratio = probs(actions[i]) / old_probs[i];
            float surr1 = prob_ratio * advantages[i];
            float surr2 = std::clamp(prob_ratio, 1.0f - epsilon_, 1.0f + epsilon_) * advantages[i];
            
            // Update policy
            float policy_loss = -std::min(surr1, surr2);
            Eigen::VectorXd policy_grad = Eigen::VectorXd::Zero(action_dim_);
            policy_grad(actions[i]) = -advantages[i];
            policy_weights_ -= learning_rate * policy_grad * states[i].transpose();
            policy_bias_ -= learning_rate * policy_grad;
            
            // Update value function
            float value_pred = getValue(states[i]);
            float value_loss = 0.5f * (value_pred - returns[i]) * (value_pred - returns[i]);
            Eigen::VectorXd value_grad = (value_pred - returns[i]) * states[i];
            value_weights_ -= learning_rate * value_grad.transpose();
            value_bias_ -= learning_rate * (value_pred - returns[i]) * Eigen::VectorXd::Ones(1);
        }
    }

private:
    Eigen::VectorXd softmax(const Eigen::VectorXd& x) const {
        Eigen::VectorXd exp_x = x.array().exp();
        return exp_x / exp_x.sum();
    }

    void computeAdvantagesAndReturns(const std::vector<float>& rewards,
                                   const std::vector<float>& values,
                                   std::vector<float>& advantages,
                                   std::vector<float>& returns) {
        advantages.resize(rewards.size());
        returns.resize(rewards.size());
        
        float next_value = 0.0f;
        float next_advantage = 0.0f;
        
        for (int i = rewards.size() - 1; i >= 0; i--) {
            float delta = rewards[i] + 0.99f * next_value - values[i];
            advantages[i] = delta + 0.95f * 0.99f * next_advantage;
            returns[i] = rewards[i] + 0.99f * next_value;
            
            next_value = values[i];
            next_advantage = advantages[i];
        }
    }

    int state_dim_;
    int action_dim_;
    std::mt19937 rng_;
    
    // Policy network parameters
    Eigen::MatrixXd policy_weights_;
    Eigen::VectorXd policy_bias_;
    
    // Value network parameters
    Eigen::MatrixXd value_weights_;
    Eigen::VectorXd value_bias_;
    
    // PPO hyperparameters
    float epsilon_;
    float c1_;
    float c2_;
}; 
#include "Environment.hpp"
#include "PPOAgent.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

// Function to get valid actions
std::vector<int> getValidActions(const Environment& env) {
    std::vector<int> valid_actions;
    for (int i = 0; i < 3; i++) {  // 3 card types
        if (env.isValidAction(i)) {
            valid_actions.push_back(i);
        }
    }
    return valid_actions;
}

// Function to print game state
void printGameState(const Environment& env, const PPOAgent& agent) {
    Eigen::VectorXd state = env.getState();
    Eigen::VectorXd probs = agent.getPolicyProbs(state);
    
    std::cout << "\nCurrent State:\n";
    std::cout << "Last played: ";
    if (state.segment(0, 3).sum() == 0) {
        std::cout << "None";
    } else {
        for (int i = 0; i < 3; i++) {
            if (state(i) > 0) {
                std::cout << Card(static_cast<CardType>(i)).getName();
            }
        }
    }
    std::cout << "\n";
    
    std::cout << "Action probabilities:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << Card(static_cast<CardType>(i)).getName() << ": "
                  << std::fixed << std::setprecision(3) << probs(i) << "\n";
    }
}

int main() {
    Environment env;
    PPOAgent agent(9, 3);  // 9 state dimensions, 3 actions
    
    const int num_episodes = 1000;
    const int episodes_per_update = 10;
    
    std::vector<Eigen::VectorXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    
    float total_reward = 0.0f;
    
    std::cout << "Starting training...\n";
    
    for (int episode = 0; episode < num_episodes; episode++) {
        env.reset();
        float episode_reward = 0.0f;
        
        while (true) {
            Eigen::VectorXd state = env.getState();
            std::vector<int> valid_actions = getValidActions(env);
            
            // Get action from policy
            int action = agent.sampleAction(state, valid_actions);
            float value = agent.getValue(state);
            
            // Take action in environment
            auto [reward, done] = env.step(action);
            episode_reward += reward;
            
            // Store transition
            states.push_back(state);
            actions.push_back(action);
            rewards.push_back(reward);
            values.push_back(value);
            
            // Visualize every 100 episodes
            if (episode % 100 == 0) {
                printGameState(env, agent);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            
            if (done) break;
        }
        
        total_reward += episode_reward;
        
        // Update policy every episodes_per_update episodes
        if ((episode + 1) % episodes_per_update == 0) {
            agent.update(states, actions, rewards, values);
            states.clear();
            actions.clear();
            rewards.clear();
            values.clear();
            
            float avg_reward = total_reward / episodes_per_update;
            std::cout << "Episode " << episode + 1 << ", Average Reward: "
                      << std::fixed << std::setprecision(3) << avg_reward << "\n";
            total_reward = 0.0f;
        }
    }
    
    std::cout << "\nTraining completed!\n";
    
    // Play a few games to demonstrate learned behavior
    std::cout << "\nPlaying demonstration games...\n";
    for (int i = 0; i < 3; i++) {
        env.reset();
        std::cout << "\nGame " << i + 1 << ":\n";
        
        while (true) {
            Eigen::VectorXd state = env.getState();
            std::vector<int> valid_actions = getValidActions(env);
            
            printGameState(env, agent);
            
            int action = agent.sampleAction(state, valid_actions);
            auto [reward, done] = env.step(action);
            
            std::cout << "Agent played: " << Card(static_cast<CardType>(action)).getName()
                      << ", Reward: " << reward << "\n";
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            if (done) break;
        }
    }
    
    return 0;
} 
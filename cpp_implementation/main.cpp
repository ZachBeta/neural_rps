#include "Environment.hpp"
#include "PPOAgent.hpp"
#include "NetworkVisualizer.hpp"
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
void printGameState(const Environment& env, const PPOAgent& agent, bool to_file = false) {
    Eigen::VectorXd state = env.getState();
    Eigen::VectorXd probs = agent.getPolicyProbs(state);
    
    std::vector<std::string> action_labels = {"Warrior", "Mage", "Archer"};
    NetworkVisualizer::visualizeActionProbs(probs, action_labels, to_file);
}

int main() {
    // Initialize file output
    NetworkVisualizer::initFileOutput("../cpp_demo_output.txt");
    
    std::cout << "Running C++ Neural RPS Demo - Output will be saved to cpp_demo_output.txt" << std::endl;
    
    Environment env;
    PPOAgent agent(9, 3);  // 9 state dimensions, 3 actions
    
    // Visualize network architecture
    std::vector<int> layer_sizes = {9, 3};  // Input layer and output layer
    std::vector<std::string> layer_names = {"Input", "Output"};
    NetworkVisualizer::visualizeArchitecture(layer_sizes, layer_names, true);
    
    const int num_episodes = 1000;
    const int episodes_per_update = 10;
    
    std::vector<Eigen::VectorXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> episode_rewards;  // Track rewards for visualization
    
    float total_reward = 0.0f;
    
    std::cout << "Starting training...\n";
    NetworkVisualizer::getOutputFile() << "Starting training...\n";
    
    // Visualize initial weights
    std::vector<std::string> input_labels = {
        "LastW", "LastM", "LastA",
        "HandW", "HandM", "HandA",
        "OppW", "OppM", "OppA"
    };
    std::vector<std::string> output_labels = {"Warrior", "Mage", "Archer"};
    NetworkVisualizer::visualizeWeights(agent.getPolicyWeights(), input_labels, output_labels, true);
    
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
                printGameState(env, agent, true);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            
            if (done) break;
        }
        
        total_reward += episode_reward;
        episode_rewards.push_back(episode_reward);
        
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
            NetworkVisualizer::getOutputFile() << "Episode " << episode + 1 << ", Average Reward: "
                                              << std::fixed << std::setprecision(3) << avg_reward << "\n";
            
            // Visualize weights and training progress
            if ((episode + 1) % 100 == 0) {
                NetworkVisualizer::visualizeWeights(agent.getPolicyWeights(), input_labels, output_labels, true);
                NetworkVisualizer::visualizeTrainingProgress(episode_rewards, 100, true);
            }
            
            total_reward = 0.0f;
        }
    }
    
    std::cout << "\nTraining completed!\n";
    NetworkVisualizer::getOutputFile() << "\nTraining completed!\n";
    
    // Final visualization of weights and training progress
    NetworkVisualizer::visualizeWeights(agent.getPolicyWeights(), input_labels, output_labels, true);
    NetworkVisualizer::visualizeTrainingProgress(episode_rewards, 100, true);
    
    // Play a few games to demonstrate learned behavior
    std::cout << "\nPlaying demonstration games...\n";
    NetworkVisualizer::getOutputFile() << "\nPlaying demonstration games...\n";
    
    for (int i = 0; i < 3; i++) {
        env.reset();
        std::cout << "\nGame " << i + 1 << ":\n";
        NetworkVisualizer::getOutputFile() << "\nGame " << i + 1 << ":\n";
        
        while (true) {
            Eigen::VectorXd state = env.getState();
            std::vector<int> valid_actions = getValidActions(env);
            
            printGameState(env, agent, true);
            
            int action = agent.sampleAction(state, valid_actions);
            auto [reward, done] = env.step(action);
            
            std::string move = "Agent played: " + Card(static_cast<CardType>(action)).getName() +
                              ", Reward: " + std::to_string(reward) + "\n";
            std::cout << move;
            NetworkVisualizer::getOutputFile() << move;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            if (done) break;
        }
    }
    
    // Close file output
    NetworkVisualizer::closeFileOutput();
    
    return 0;
} 
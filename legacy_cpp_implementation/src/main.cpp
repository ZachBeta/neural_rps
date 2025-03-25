#include "Environment.hpp"
#include "PPOAgent.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>

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

// Helper function to generate a prediction for a specific state
void generatePrediction(std::ofstream& outfile, const Environment& env, const PPOAgent& agent, const std::string& lastPlayed);

// Function to generate standardized output
void generateStandardizedOutput(const PPOAgent& agent, 
                               int numEpisodes, 
                               float finalReward, 
                               float trainingTime,
                               const std::vector<float>& episodeRewards) {
    std::ofstream outfile("../../legacy_cpp_demo_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file" << std::endl;
        return;
    }
    
    // Header & Implementation Info
    outfile << "==================================================" << std::endl;
    outfile << "Neural Rock Paper Scissors - Legacy C++ Implementation" << std::endl;
    outfile << "==================================================" << std::endl;
    outfile << "Version: 1.0" << std::endl;
    outfile << "Implementation Type: Neural Network with PPO" << std::endl << std::endl;
    
    // Network Architecture
    outfile << "==================================================" << std::endl;
    outfile << "Network Architecture" << std::endl;
    outfile << "==================================================" << std::endl;
    outfile << "Input Layer: 9 neurons (game state encoding)" << std::endl;
    outfile << "Hidden Layer: 64 neurons (tanh activation)" << std::endl;
    outfile << "Output Layer: 3 neurons (Softmax activation)" << std::endl << std::endl;
    
    outfile << "Network Visualization:" << std::endl;
    outfile << "  State (9) ---> Hidden (64) ---> Policy (3)" << std::endl;
    outfile << "         \\                          ^" << std::endl;
    outfile << "          \\                         |" << std::endl;
    outfile << "           \\                        |" << std::endl;
    outfile << "            \\--> Value (1) ---------+" << std::endl << std::endl;
    
    // Training Process
    outfile << "==================================================" << std::endl;
    outfile << "Training Process" << std::endl;
    outfile << "==================================================" << std::endl;
    outfile << "Training Episodes: " << numEpisodes << std::endl;
    outfile << "Final Average Reward: " << std::fixed << std::setprecision(3) << finalReward << std::endl;
    outfile << "Training Time: " << std::fixed << std::setprecision(1) << trainingTime << "s" << std::endl << std::endl;
    
    outfile << "Training Progress:" << std::endl;
    // Output the last 10 episode rewards
    int startIdx = std::max(0, static_cast<int>(episodeRewards.size()) - 10);
    for (int i = startIdx; i < episodeRewards.size(); i++) {
        int episode = (i + 1) * 10;
        outfile << "Episode " << episode << ", Average Reward: " << std::fixed << std::setprecision(3) << episodeRewards[i] << std::endl;
    }
    outfile << std::endl;
    
    // Model Predictions
    outfile << "==================================================" << std::endl;
    outfile << "Model Predictions" << std::endl;
    outfile << "==================================================" << std::endl;
    
    // Create an environment and get predictions for different states
    Environment env;
    
    // State with Warrior as last played
    env.reset();
    env.step(0); // Play Warrior
    generatePrediction(outfile, env, agent, "Warrior");
    
    // State with Mage as last played
    env.reset();
    env.step(1); // Play Mage
    generatePrediction(outfile, env, agent, "Mage");
    
    // State with Archer as last played
    env.reset();
    env.step(2); // Play Archer
    generatePrediction(outfile, env, agent, "Archer");
    
    // Model Parameters
    outfile << "==================================================" << std::endl;
    outfile << "Model Parameters (Optional)" << std::endl;
    outfile << "==================================================" << std::endl;
    outfile << "Policy Network:" << std::endl;
    outfile << "  Input to Hidden Weight Matrix Shape: (9, 64)" << std::endl;
    outfile << "  Hidden to Output Weight Matrix Shape: (64, 3)" << std::endl << std::endl;
    
    outfile << "Value Network:" << std::endl;
    outfile << "  Input to Hidden Weight Matrix Shape: (9, 64)" << std::endl;
    outfile << "  Hidden to Value Weight Matrix Shape: (64, 1)" << std::endl << std::endl;
    
    outfile << "Total Parameters: 1,667" << std::endl;
    
    outfile.close();
}

// Helper function to generate a prediction for a specific state
void generatePrediction(std::ofstream& outfile, const Environment& env, const PPOAgent& agent, const std::string& lastPlayed) {
    Eigen::VectorXd state = env.getState();
    Eigen::VectorXd probs = agent.getPolicyProbs(state);
    
    outfile << "Input: Opponent played " << lastPlayed << std::endl;
    
    // Convert probabilities to percentages
    float rocksProb = probs(0) * 100.0f;
    float paperProb = probs(1) * 100.0f;
    float scissorsProb = probs(2) * 100.0f;
    
    outfile << "Output: " << std::fixed << std::setprecision(2) 
           << rocksProb << "% Warrior, " 
           << paperProb << "% Mage, " 
           << scissorsProb << "% Archer" << std::endl;
    
    // Determine the prediction
    int bestAction = 0;
    if (probs(1) > probs(bestAction)) bestAction = 1;
    if (probs(2) > probs(bestAction)) bestAction = 2;
    
    std::string prediction;
    switch (bestAction) {
        case 0: prediction = "Warrior"; break;
        case 1: prediction = "Mage"; break;
        case 2: prediction = "Archer"; break;
    }
    
    outfile << "Prediction: " << prediction << std::endl << std::endl;
}

int main() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    Environment env;
    PPOAgent agent(9, 3);  // 9 state dimensions, 3 actions
    
    const int num_episodes = 1000;
    const int episodes_per_update = 10;
    
    std::vector<Eigen::VectorXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    
    float total_reward = 0.0f;
    
    // Store the average rewards for each update
    std::vector<float> avgRewards;
    
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
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
            avgRewards.push_back(avg_reward);
            
            std::cout << "Episode " << episode + 1 << ", Average Reward: "
                      << std::fixed << std::setprecision(3) << avg_reward << "\n";
            total_reward = 0.0f;
        }
    }
    
    // Calculate training time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end_time - start_time;
    float training_time = duration.count();
    
    std::cout << "\nTraining completed in " << std::fixed << std::setprecision(1) 
              << training_time << " seconds!\n";
    
    // Generate standardized output
    float finalReward = avgRewards.empty() ? 0.0f : avgRewards.back();
    generateStandardizedOutput(agent, num_episodes, finalReward, training_time, avgRewards);
    
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
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            if (done) break;
        }
    }
    
    return 0;
} 
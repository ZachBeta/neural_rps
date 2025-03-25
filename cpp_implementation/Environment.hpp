#pragma once

#include <vector>
#include <array>
#include <random>
#include <tuple>
#include <Eigen/Dense>

enum class CardType {
    WARRIOR = 0,
    MAGE = 1,
    ARCHER = 2
};

class Card {
public:
    explicit Card(CardType type) : type_(type) {}
    
    CardType getType() const { return type_; }
    
    std::string getName() const {
        switch (type_) {
            case CardType::WARRIOR: return "Warrior";
            case CardType::MAGE: return "Mage";
            case CardType::ARCHER: return "Archer";
            default: return "Unknown";
        }
    }
    
private:
    CardType type_;
};

class Environment {
public:
    Environment();
    
    void reset();
    std::tuple<float, bool> step(int action);
    Eigen::VectorXd getState() const;
    bool isValidAction(int action) const;
    
private:
    std::array<int, 3> hand_; // Count of each card type in hand
    CardType last_player_card_{CardType::WARRIOR};
    CardType last_opponent_card_{CardType::WARRIOR};
    std::mt19937 rng_;
};

inline Environment::Environment() : rng_(std::random_device{}()) {
    reset();
}

inline void Environment::reset() {
    // Start with 3 of each card type
    hand_ = {3, 3, 3};
    last_player_card_ = CardType::WARRIOR;
    last_opponent_card_ = CardType::WARRIOR;
}

inline std::tuple<float, bool> Environment::step(int action) {
    // Validate action
    if (!isValidAction(action)) {
        return {-5.0f, true}; // Invalid action, big penalty and terminate
    }
    
    // Update player's card and hand
    last_player_card_ = static_cast<CardType>(action);
    hand_[action]--;
    
    // Opponent plays random card
    std::uniform_int_distribution<int> dist(0, 2);
    int opponent_action = dist(rng_);
    last_opponent_card_ = static_cast<CardType>(opponent_action);
    
    // Calculate reward based on Rock-Paper-Scissors rules
    float reward = 0.0f;
    if (last_player_card_ == last_opponent_card_) {
        reward = 0.0f; // Draw
    } else if ((last_player_card_ == CardType::WARRIOR && last_opponent_card_ == CardType::ARCHER) ||
               (last_player_card_ == CardType::MAGE && last_opponent_card_ == CardType::WARRIOR) ||
               (last_player_card_ == CardType::ARCHER && last_opponent_card_ == CardType::MAGE)) {
        reward = 1.0f; // Win
    } else {
        reward = -1.0f; // Lose
    }
    
    // Check if game is over (no cards left)
    bool done = (hand_[0] + hand_[1] + hand_[2] == 0);
    
    return {reward, done};
}

inline Eigen::VectorXd Environment::getState() const {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(9);
    
    // One-hot encode last player card
    state[static_cast<int>(last_player_card_)] = 1.0;
    
    // Normalized hand representation
    for (int i = 0; i < 3; i++) {
        state[i + 3] = static_cast<float>(hand_[i]) / 3.0f;
    }
    
    // One-hot encode last opponent card
    state[static_cast<int>(last_opponent_card_) + 6] = 1.0;
    
    return state;
}

inline bool Environment::isValidAction(int action) const {
    if (action < 0 || action >= 3) return false;
    return hand_[action] > 0;
} 
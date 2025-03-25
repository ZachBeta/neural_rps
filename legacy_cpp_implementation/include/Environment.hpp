#pragma once

#include "Card.hpp"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <algorithm>

class Environment {
public:
    Environment() : rng_(std::random_device{}()) {
        reset();
    }

    // Reset the environment to initial state
    void reset() {
        // Clear history and create new hands
        history_.clear();
        current_player_ = 0;
        
        // Initialize hands for both players
        player_hands_[0] = generateHand();
        player_hands_[1] = generateHand();
    }

    // Get the current state as a vector (for neural network input)
    Eigen::VectorXd getState() const {
        Eigen::VectorXd state = Eigen::VectorXd::Zero(9);  // 3 cards * (last_play + hand_p1 + hand_p2)
        
        // Encode last played card (if any)
        if (!history_.empty()) {
            int last_card_idx = static_cast<int>(history_.back().getType());
            state(last_card_idx) = 1.0;
        }

        // Get the hands from current player's perspective
        const auto& current_hand = player_hands_[current_player_];
        const auto& opponent_hand = player_hands_[1 - current_player_];

        // Encode current player's hand (positions 3-5)
        for (const auto& card : current_hand) {
            int card_idx = static_cast<int>(card.getType()) + 3;
            state(card_idx) = 1.0;
        }

        // Encode opponent's hand (positions 6-8)
        for (const auto& card : opponent_hand) {
            int card_idx = static_cast<int>(card.getType()) + 6;
            state(card_idx) = 1.0;
        }

        return state;
    }

    // Take an action and return (reward, is_terminal)
    std::pair<float, bool> step(int action) {
        // Convert action to card
        Card played_card(static_cast<CardType>(action));
        
        // Verify the action is valid
        if (!isValidAction(action)) {
            return {-1.0f, true};  // Invalid action
        }
        
        // Store the current player for reward calculation
        int playing_player = current_player_;
        
        // Remove card from hand
        removeCardFromHand(played_card);
        
        float reward = 0.0;
        
        // If there's a previous play, determine winner
        if (!history_.empty()) {
            Card last_card = history_.back();
            if (played_card.beats(last_card)) {
                reward = 1.0;
            } else if (last_card.beats(played_card)) {
                reward = -1.0;
            }
        }
        
        // Add card to history
        history_.push_back(played_card);
        
        // Switch players
        current_player_ = 1 - current_player_;
        
        // Game ends when both players have played all cards
        bool is_terminal = history_.size() >= 6;  // 3 cards each
        
        return {reward, is_terminal};
    }

    bool isValidAction(int action) const {
        if (action < 0 || action > 2) return false;
        
        CardType type = static_cast<CardType>(action);
        return std::find_if(player_hands_[current_player_].begin(),
                          player_hands_[current_player_].end(),
                          [type](const Card& c) { return c.getType() == type; })
               != player_hands_[current_player_].end();
    }

private:
    std::vector<Card> generateHand() {
        std::vector<Card> hand;
        hand.push_back(Card(CardType::WARRIOR));
        hand.push_back(Card(CardType::MAGE));
        hand.push_back(Card(CardType::ARCHER));
        return hand;
    }

    void removeCardFromHand(const Card& card) {
        auto& hand = player_hands_[current_player_];
        auto it = std::find_if(hand.begin(), hand.end(),
                             [&card](const Card& c) { return c.getType() == card.getType(); });
        
        if (it != hand.end()) {
            hand.erase(it);
        }
    }

    std::mt19937 rng_;
    std::vector<Card> history_;
    std::vector<Card> player_hands_[2];
    int current_player_ = 0;
}; 
#pragma once

#include <string>
#include <vector>

enum class CardType {
    WARRIOR = 0,  // Beats Mage
    MAGE = 1,     // Beats Archer
    ARCHER = 2    // Beats Warrior
};

class Card {
public:
    Card(CardType type) : type_(type) {}

    CardType getType() const { return type_; }
    
    // Returns true if this card beats the other card
    bool beats(const Card& other) const {
        switch (type_) {
            case CardType::WARRIOR:
                return other.type_ == CardType::ARCHER;
            case CardType::MAGE:
                return other.type_ == CardType::WARRIOR;
            case CardType::ARCHER:
                return other.type_ == CardType::MAGE;
        }
        return false;
    }

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
#include "Card.hpp"
#include <gtest/gtest.h>

TEST(CardTest, CardCreation) {
    Card warrior(CardType::WARRIOR);
    Card mage(CardType::MAGE);
    Card archer(CardType::ARCHER);
    
    EXPECT_EQ(warrior.getName(), "Warrior");
    EXPECT_EQ(mage.getName(), "Mage");
    EXPECT_EQ(archer.getName(), "Archer");
}

TEST(CardTest, CardRelationships) {
    Card warrior(CardType::WARRIOR);
    Card mage(CardType::MAGE);
    Card archer(CardType::ARCHER);
    
    // Test the circular relationship
    EXPECT_TRUE(warrior.beats(archer));   // Warrior beats Archer
    EXPECT_TRUE(archer.beats(mage));      // Archer beats Mage
    EXPECT_TRUE(mage.beats(warrior));     // Mage beats Warrior
    
    // Test inverse relationships
    EXPECT_FALSE(archer.beats(warrior));  // Archer doesn't beat Warrior
    EXPECT_FALSE(mage.beats(archer));     // Mage doesn't beat Archer
    EXPECT_FALSE(warrior.beats(mage));    // Warrior doesn't beat Mage
}

TEST(CardTest, SelfComparison) {
    Card warrior1(CardType::WARRIOR);
    Card warrior2(CardType::WARRIOR);
    
    // A card should not beat itself
    EXPECT_FALSE(warrior1.beats(warrior2));
    EXPECT_FALSE(warrior2.beats(warrior1));
}

TEST(CardTest, TypeConsistency) {
    Card warrior(CardType::WARRIOR);
    EXPECT_EQ(warrior.getType(), CardType::WARRIOR);
    EXPECT_EQ(static_cast<int>(warrior.getType()), 0);
    
    Card mage(CardType::MAGE);
    EXPECT_EQ(mage.getType(), CardType::MAGE);
    EXPECT_EQ(static_cast<int>(mage.getType()), 1);
    
    Card archer(CardType::ARCHER);
    EXPECT_EQ(archer.getType(), CardType::ARCHER);
    EXPECT_EQ(static_cast<int>(archer.getType()), 2);
} 
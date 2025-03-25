#include "Environment.hpp"
#include <gtest/gtest.h>

TEST(EnvironmentTest, Initialization) {
    Environment env;
    Eigen::VectorXd state = env.getState();
    
    // Initial state should be 9-dimensional
    EXPECT_EQ(state.size(), 9);
    
    // No last played card in initial state
    EXPECT_EQ(state.segment(0, 3).sum(), 0);
    
    // Each player should have 3 cards
    EXPECT_EQ(state.segment(3, 3).sum(), 3);  // Current player's hand
    EXPECT_EQ(state.segment(6, 3).sum(), 3);  // Opponent's hand
}

TEST(EnvironmentTest, ValidActions) {
    Environment env;
    
    // Initially all actions should be valid
    EXPECT_TRUE(env.isValidAction(0));  // Warrior
    EXPECT_TRUE(env.isValidAction(1));  // Mage
    EXPECT_TRUE(env.isValidAction(2));  // Archer
    
    // Invalid action indices should return false
    EXPECT_FALSE(env.isValidAction(-1));
    EXPECT_FALSE(env.isValidAction(3));
    
    // Play Warrior (action 0)
    auto [reward, done] = env.step(0);
    Eigen::VectorXd state = env.getState();
    
    // The played card should be in history
    EXPECT_EQ(state(0), 1.0);  // Warrior in history
    
    // Check that the current player (now player 2) has all their cards
    EXPECT_EQ(state.segment(3, 3).sum(), 3.0);
}

TEST(EnvironmentTest, GameFlow) {
    Environment env;
    std::vector<std::pair<float, bool>> results;
    
    // Play all cards and verify the game state after each move
    for (int i = 0; i < 6; i++) {  // 3 cards per player
        Eigen::VectorXd state = env.getState();
        
        // Current player should always have the correct number of cards
        int expected_cards = 3 - (i / 2);  // Each player loses a card every 2 moves
        EXPECT_EQ(state.segment(3, 3).sum(), expected_cards);
        
        // Make a valid move
        for (int action = 0; action < 3; action++) {
            if (env.isValidAction(action)) {
                results.push_back(env.step(action));
                break;
            }
        }
    }
    
    // Game should end after 6 moves (3 per player)
    EXPECT_EQ(results.size(), 6);
    EXPECT_TRUE(results.back().second);  // Last move should end the game
    
    // Final state should have no cards in current player's hand
    Eigen::VectorXd final_state = env.getState();
    EXPECT_EQ(final_state.segment(3, 3).sum(), 0);
}

TEST(EnvironmentTest, Rewards) {
    Environment env;
    
    // Play Warrior (beats Archer)
    auto [r1, d1] = env.step(0);  // Player 1 plays Warrior
    auto [r2, d2] = env.step(2);  // Player 2 plays Archer
    
    // Player 2 should lose (negative reward) because Warrior beats Archer
    EXPECT_LT(r2, 0);
    
    // Reset and try different combination
    env.reset();
    
    // Play Mage (beats Warrior)
    auto [r3, d3] = env.step(0);  // Player 1 plays Warrior
    auto [r4, d4] = env.step(1);  // Player 2 plays Mage
    
    // Player 2 should win (positive reward) because Mage beats Warrior
    EXPECT_GT(r4, 0);
}

TEST(EnvironmentTest, StateEncoding) {
    Environment env;
    
    // Get initial state
    Eigen::VectorXd state1 = env.getState();
    EXPECT_EQ(state1.segment(3, 3).sum(), 3.0);  // First player has 3 cards
    
    // Play Warrior (action 0)
    env.step(0);
    Eigen::VectorXd state2 = env.getState();
    
    // The played card should now be in the history
    EXPECT_EQ(state2(0), 1.0);  // Warrior in history
    
    // Second player should now be current and have all their cards
    EXPECT_EQ(state2.segment(3, 3).sum(), 3.0);  // Current player (P2) has 3 cards
    
    // First player should now be opponent and have 2 cards
    EXPECT_EQ(state2.segment(6, 3).sum(), 2.0);  // Opponent (P1) has 2 cards
} 
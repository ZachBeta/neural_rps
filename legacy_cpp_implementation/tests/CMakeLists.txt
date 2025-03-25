# Add test executables
add_executable(test_card test_card.cpp)
add_executable(test_environment test_environment.cpp)
add_executable(test_neural_network test_neural_network.cpp ../src/NeuralNetwork.cpp)
add_executable(test_ppo_agent test_ppo_agent.cpp)

# Link tests with GTest
target_link_libraries(test_card GTest::gtest_main)
target_link_libraries(test_environment GTest::gtest_main Eigen3::Eigen)
target_link_libraries(test_neural_network GTest::gtest_main Eigen3::Eigen)
target_link_libraries(test_ppo_agent GTest::gtest_main Eigen3::Eigen)

# Register tests
add_test(NAME CardTest COMMAND test_card)
add_test(NAME EnvironmentTest COMMAND test_environment)
add_test(NAME NeuralNetworkTest COMMAND test_neural_network)
add_test(NAME PPOAgentTest COMMAND test_ppo_agent) 
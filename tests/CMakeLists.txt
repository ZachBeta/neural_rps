add_executable(test_card test_card.cpp)
target_link_libraries(test_card
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
)
add_test(NAME test_card COMMAND test_card)

add_executable(test_environment test_environment.cpp)
target_link_libraries(test_environment
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
)
add_test(NAME test_environment COMMAND test_environment)

add_executable(test_ppo_agent test_ppo_agent.cpp)
target_link_libraries(test_ppo_agent
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
)
add_test(NAME test_ppo_agent COMMAND test_ppo_agent) 
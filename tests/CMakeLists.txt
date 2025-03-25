add_executable(neural_rps_test
    test_neural_network.cpp
)

target_link_libraries(neural_rps_test
    neural_rps_lib
    GTest::GTest
    GTest::Main
)

add_test(NAME neural_rps_test COMMAND neural_rps_test) 
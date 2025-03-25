add_executable(legacy_neural_rps
    main.cpp
    NeuralNetwork.cpp
)

target_link_libraries(legacy_neural_rps
    Eigen3::Eigen
) 
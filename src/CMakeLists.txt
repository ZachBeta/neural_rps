add_library(neural_rps_lib
    NeuralNetwork.cpp
)

add_executable(neural_rps
    main.cpp
)

target_link_libraries(neural_rps
    neural_rps_lib
    Eigen3::Eigen
) 
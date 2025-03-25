cmake_minimum_required(VERSION 3.10)
project(neural_rps)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(WARNING "Eigen3 not found! This implementation requires Eigen3 library.")
    message(WARNING "You can install it with: brew install eigen (on macOS) or apt-get install libeigen3-dev (on Ubuntu)")
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add a simpler program that just outputs a demo message
add_executable(neural_rps_demo simple_demo.cpp)

# Add the full neural network implementation
add_executable(neural_rps_full main.cpp NeuralNetwork.cpp)

# Enable clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
else()
    message(STATUS "clang-tidy not found!")
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif() 
cmake_minimum_required(VERSION 3.10)
project(neural_rps)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Option to enable detailed network visualization
option(DETAILED_VIZ "Enable detailed network visualization" OFF)
if(DETAILED_VIZ)
    add_definitions(-DDETAILED_VIZ)
endif()

# Option to build the RPS agent
option(BUILD_AGENT "Build the RPS agent executable" OFF)

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

# Create the main executable
add_executable(neural_rps main.cpp NeuralNetwork.cpp)

# Enable testing
enable_testing()

# Basic test to verify the neural network learns
add_executable(test_learning test_learning.cpp NeuralNetwork.cpp)
add_test(NAME TestLearning COMMAND test_learning)

# Test for network serialization
add_executable(test_serialization test_serialization.cpp NeuralNetwork.cpp)
add_test(NAME TestSerialization COMMAND test_serialization)

# Simple demo executable
add_executable(simple_demo simple_demo.cpp NeuralNetwork.cpp)

# RPS agent executable (only built when BUILD_AGENT=ON)
if(BUILD_AGENT)
    add_executable(rps_agent rps_agent.cpp NeuralNetwork.cpp)
endif()

# Installation targets
install(TARGETS neural_rps simple_demo DESTINATION bin)
install(FILES NeuralNetwork.hpp DESTINATION include)

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
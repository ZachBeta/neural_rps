cmake_minimum_required(VERSION 3.10)
project(neural_rps)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(neural_rps main.cpp NeuralNetwork.cpp)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR})

# Enable clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
else()
    message(STATUS "clang-tidy not found!")
endif()

# Find Eigen3 (Optional)
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(WARNING "Eigen3 not found! This implementation requires Eigen3 library.")
    message(WARNING "You can install it with: brew install eigen (on macOS) or apt-get install libeigen3-dev (on Ubuntu)")
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif() 
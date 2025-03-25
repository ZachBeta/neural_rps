cmake_minimum_required(VERSION 3.14)
project(legacy_neural_rps VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(GTest QUIET)

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add subdirectories
add_subdirectory(src)

# Optionally add tests if GTest is found
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif() 
cmake_minimum_required(VERSION 3.14)
project(neural_rps VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
else()
    message(STATUS "clang-tidy not found!")
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif() 
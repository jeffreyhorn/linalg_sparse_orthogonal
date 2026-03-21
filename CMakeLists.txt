cmake_minimum_required(VERSION 3.14)
project(sparse_lu_orthogonal C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wdouble-promotion
    -Wformat=2
    -Wno-unused-parameter
)

# Optional sanitizers (enable with -DSANITIZE=ON)
option(SANITIZE "Enable address and undefined behavior sanitizers" OFF)
if(SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Library
add_library(sparse_lu_ortho STATIC
    src/sparse_types.c
    src/sparse_matrix.c
    src/sparse_lu.c
    src/sparse_vector.c
)

target_include_directories(sparse_lu_ortho
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(sparse_lu_ortho PUBLIC m)

# Tests
enable_testing()

# Helper function to add a test executable
function(add_sparse_test TEST_NAME)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.c)
    target_link_libraries(${TEST_NAME} PRIVATE sparse_lu_ortho)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Test executables
add_sparse_test(test_sparse_matrix)
add_sparse_test(test_sparse_lu)
add_sparse_test(test_sparse_io)
add_sparse_test(test_known_matrices)
add_sparse_test(test_sparse_vector)
add_sparse_test(test_edge_cases)
add_sparse_test(test_integration)

# Benchmarks (not part of ctest)
add_executable(bench_main benchmarks/bench_main.c)
target_link_libraries(bench_main PRIVATE sparse_lu_ortho)

add_executable(bench_scaling benchmarks/bench_scaling.c)
target_link_libraries(bench_scaling PRIVATE sparse_lu_ortho)

add_executable(bench_fillin benchmarks/bench_fillin.c)
target_link_libraries(bench_fillin PRIVATE sparse_lu_ortho)

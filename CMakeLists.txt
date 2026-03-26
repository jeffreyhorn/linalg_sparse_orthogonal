cmake_minimum_required(VERSION 3.14)
project(linalg_sparse_orthogonal C)

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

# Optional sanitizers
# Usage: -DSANITIZE=asan | -DSANITIZE=ubsan | -DSANITIZE=all | -DSANITIZE=ON (same as all)
set(SANITIZE "" CACHE STRING "Sanitizer mode: asan, ubsan, all, or ON")
if(SANITIZE STREQUAL "asan")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(SANITIZE STREQUAL "ubsan")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
elseif(SANITIZE STREQUAL "all" OR SANITIZE STREQUAL "ON")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Library
add_library(sparse_lu_ortho STATIC
    src/sparse_types.c
    src/sparse_matrix.c
    src/sparse_lu.c
    src/sparse_vector.c
    src/sparse_reorder.c
    src/sparse_cholesky.c
    src/sparse_csr.c
    src/sparse_iterative.c
    src/sparse_ilu.c
)

# Optional OpenMP support for parallel SpMV: cmake -DSPARSE_OPENMP=ON ..
option(SPARSE_OPENMP "Enable OpenMP parallelization for SpMV" OFF)
if(SPARSE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(sparse_lu_ortho PUBLIC SPARSE_OPENMP)
    target_link_libraries(sparse_lu_ortho PUBLIC OpenMP::OpenMP_C)
endif()

# Optional mutex support: cmake -DSPARSE_MUTEX=ON ..
option(SPARSE_MUTEX "Enable per-matrix mutex for concurrent insert/remove" OFF)
if(SPARSE_MUTEX)
    find_package(Threads REQUIRED)
    target_compile_definitions(sparse_lu_ortho PUBLIC SPARSE_MUTEX)
    target_link_libraries(sparse_lu_ortho PUBLIC Threads::Threads)
endif()

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
add_sparse_test(test_sparse_arith)
add_sparse_test(test_suitesparse)
add_sparse_test(test_iterative)
add_sparse_test(test_ilu)
add_sparse_test(test_omp)
add_sparse_test(test_sprint5_integration)

# Benchmarks (not part of ctest)
add_executable(bench_main benchmarks/bench_main.c)
target_link_libraries(bench_main PRIVATE sparse_lu_ortho)

add_executable(bench_scaling benchmarks/bench_scaling.c)
target_link_libraries(bench_scaling PRIVATE sparse_lu_ortho)

add_executable(bench_fillin benchmarks/bench_fillin.c)
target_link_libraries(bench_fillin PRIVATE sparse_lu_ortho)

add_executable(bench_convergence benchmarks/bench_convergence.c)
target_link_libraries(bench_convergence PRIVATE sparse_lu_ortho)

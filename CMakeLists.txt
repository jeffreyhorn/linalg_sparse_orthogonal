cmake_minimum_required(VERSION 3.14)

# Read version from single-source-of-truth VERSION file
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SPARSE_VERSION_FROM_FILE)
list(GET SPARSE_VERSION_FROM_FILE 0 SPARSE_VERSION_STRING_FILE)
string(STRIP "${SPARSE_VERSION_STRING_FILE}" SPARSE_VERSION_STRING_FILE)

project(linalg_sparse_orthogonal VERSION ${SPARSE_VERSION_STRING_FILE} LANGUAGES C)

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
    src/sparse_qr.c
    src/sparse_dense.c
    src/sparse_bidiag.c
    src/sparse_svd.c
    src/sparse_lu_csr.c
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
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link -lm on platforms that have a separate math library (not MSVC)
if(NOT MSVC)
    target_link_libraries(sparse_lu_ortho PUBLIC m)
endif()

# Tests
enable_testing()

# Helper function to add a test executable
function(add_sparse_test TEST_NAME)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.c)
    target_link_libraries(${TEST_NAME} PRIVATE sparse_lu_ortho)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
add_sparse_test(test_qr)
add_sparse_test(test_sprint6_integration)
add_sparse_test(test_dense)
add_sparse_test(test_bidiag)
add_sparse_test(test_svd)
add_sparse_test(test_sprint8_integration)
if(NOT WIN32 AND NOT MSVC)
    add_sparse_test(test_fuzz)
endif()
add_sparse_test(test_lu_csr)
add_sparse_test(test_block_solvers)
add_sparse_test(test_sprint10_integration)

# Benchmarks (not part of ctest)
add_executable(bench_main benchmarks/bench_main.c)
target_link_libraries(bench_main PRIVATE sparse_lu_ortho)

add_executable(bench_scaling benchmarks/bench_scaling.c)
target_link_libraries(bench_scaling PRIVATE sparse_lu_ortho)

add_executable(bench_fillin benchmarks/bench_fillin.c)
target_link_libraries(bench_fillin PRIVATE sparse_lu_ortho)

add_executable(bench_convergence benchmarks/bench_convergence.c)
target_link_libraries(bench_convergence PRIVATE sparse_lu_ortho)

# ─── Installation ────────────────────────────────────────────────────

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Adjust include path for installed headers (under sparse/ subdirectory)
target_include_directories(sparse_lu_ortho
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS sparse_lu_ortho
    EXPORT SparseTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sparse
    FILES_MATCHING PATTERN "*.h"
)

# CMake package config files
set(SPARSE_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/Sparse)

install(EXPORT SparseTargets
    FILE SparseTargets.cmake
    NAMESPACE Sparse::
    DESTINATION ${SPARSE_CMAKE_DIR}
)

configure_package_config_file(
    cmake/SparseConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SparseConfig.cmake
    INSTALL_DESTINATION ${SPARSE_CMAKE_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SparseConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SparseConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SparseConfigVersion.cmake
    DESTINATION ${SPARSE_CMAKE_DIR}
)

# pkg-config (set variables that match the @VAR@ placeholders in sparse.pc.in)
set(PREFIX ${CMAKE_INSTALL_PREFIX})
set(VERSION ${PROJECT_VERSION})
set(SPARSE_PC_LIBS_EXTRA "")
if(SPARSE_MUTEX)
    string(APPEND SPARSE_PC_LIBS_EXTRA " -pthread")
endif()
if(SPARSE_OPENMP)
    string(APPEND SPARSE_PC_LIBS_EXTRA " ${OpenMP_C_FLAGS}")
endif()
configure_file(sparse.pc.in ${CMAKE_CURRENT_BINARY_DIR}/sparse.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparse.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

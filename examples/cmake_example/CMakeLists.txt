cmake_minimum_required(VERSION 3.14)
project(sparse_cmake_example C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Sparse REQUIRED)

add_executable(example main.c)
target_link_libraries(example PRIVATE Sparse::sparse_lu_ortho)

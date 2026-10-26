cmake_minimum_required(VERSION 3.14)
project(sparse_installed_width_consumer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT DEFINED SPARSE_TEST_VERSION)
    message(FATAL_ERROR "SPARSE_TEST_VERSION is required")
endif()
find_package(Sparse ${SPARSE_TEST_VERSION} EXACT REQUIRED CONFIG)

add_executable(width_consumer ../../examples/cmake_example/main.c)
target_link_libraries(width_consumer PRIVATE Sparse::sparse_lu_ortho)

if(DEFINED SPARSE_TEST_CONSUMER_BITS)
    target_compile_definitions(width_consumer PRIVATE SPARSE_IDX_BITS=${SPARSE_TEST_CONSUMER_BITS})
endif()

if(DEFINED SPARSE_TEST_ARCHIVE_OVERRIDE)
    foreach(config NOCONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_property(TARGET Sparse::sparse_lu_ortho PROPERTY IMPORTED_LOCATION_${config}
            "${SPARSE_TEST_ARCHIVE_OVERRIDE}")
    endforeach()
    set_property(TARGET Sparse::sparse_lu_ortho PROPERTY IMPORTED_LOCATION
        "${SPARSE_TEST_ARCHIVE_OVERRIDE}")
endif()

project (GEORGE C CXX)
cmake_minimum_required (VERSION 2.8)
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

set (GEORGE_VERSION_MAJOR 0)
set (GEORGE_VERSION_MINOR 1)
set (GEORGE_VERSION_PATCH 0)
set (GEORGE_VERSION
     ${GEORGE_VERSION_MAJOR}.${GEORGE_VERSION_MINOR}.${GEORGE_VERSION_PATCH})
set (GEORGE_ABI_VERSION 0.1.0)

enable_testing ()

find_package (BLAS REQUIRED)
find_package (LAPACK REQUIRED)
message ("-- Found BLAS library: ${BLAS_LIBRARIES}")
message ("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
find_package (Eigen REQUIRED)
find_package (SuiteSparse REQUIRED)
if (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
    message (FATAL_ERROR "You are attempting to build on Ubuntu using a "
             "system package install of SuiteSparse 3.4.0. This package is "
             "broken and does not support the construction of shared "
             "libraries.")
endif (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

set (CMAKE_CXX_FLAGS
     "${CMAKE_CXX_FLAGS} -O4 -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")

# Includes.
include_directories ("${CMAKE_SOURCE_DIR}/cpp"
                     "${CMAKE_SOURCE_DIR}/hodlr/header"
                     ${EIGEN_INCLUDE_DIRS}
                     ${SUITESPARSE_INCLUDE_DIRS})

set (GEORGE_LIBRARY_DEPENDENCIES ${LAPACK_LIBRARIES})
list (APPEND GEORGE_LIBRARY_DEPENDENCIES ${BLAS_LIBRARIES})
list (APPEND GEORGE_LIBRARY_DEPENDENCIES ${SUITESPARSE_LIBRARIES})

add_executable (demo cpp/demo.cc)
target_link_libraries (demo ${GEORGE_LIBRARY_DEPENDENCIES})

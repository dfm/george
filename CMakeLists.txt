project (GEORGE C)
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

ENABLE_TESTING()

find_package (BLAS REQUIRED)
find_package (LAPACK REQUIRED)
message ("-- Found BLAS library: ${BLAS_LIBRARIES}")
message ("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
find_package (LBFGS REQUIRED)
find_package (SuiteSparse REQUIRED)
if (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
    message (FATAL_ERROR "You are attempting to build on Ubuntu using a "
             "system package install of SuiteSparse 3.4.0. This package is "
             "broken and does not support the construction of shared "
             "libraries.")
endif (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

# Optimizations.
if (CMAKE_COMPILER_IS_GNUCC)
    # Linux
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        if (NOT GCC_VERSION VERSION_LESS 4.2)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
        endif (NOT GCC_VERSION VERSION_LESS 4.2)
    endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Mac OS X
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
        # Use of -fast only applicable for Apple's GCC
        # Assume this is being used if GCC version < 4.3 on OSX
        execute_process (COMMAND ${CMAKE_C_COMPILER}
                         ARGS ${CMAKE_C_COMPILER_ARG1} -dumpversion
                         OUTPUT_VARIABLE GCC_VERSION
                         OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (GCC_VERSION VERSION_LESS 4.3)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fast")
        endif (GCC_VERSION VERSION_LESS 4.3)
    endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
endif (CMAKE_COMPILER_IS_GNUCC)

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    # Use of -O4 requires use of gold linker & LLVM-gold plugin, which might
    # well not be present / in use and without which files will compile, but
    # not link ('file not recognized') so explicitly check for support
    include (CheckCCompilerFlag)
    check_c_compiler_flag ("-O4" HAVE_LTO_SUPPORT)
    if (HAVE_LTO_SUPPORT)
        message (STATUS "Enabling link-time optimization (-O4)")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4")
    else ()
        message (STATUS
                 "Compiler/linker does not support link-time optimization (-O4), disabling.")
    endif (HAVE_LTO_SUPPORT)
endif ()

# GCC is not strict enough by default, so enable most of the warnings.
if (UNIX)
    set (CMAKE_C_FLAGS
         "${CMAKE_C_FLAGS} -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
endif (UNIX)

# Includes.
include_directories ("${CMAKE_BINARY_DIR}/george"
                     ${SUITESPARSE_INCLUDE_DIRS}
                     ${LBFGS_INCLUDE_DIR})

# Add the source directory.
add_subdirectory (george)

# ===========================================================================
#   PYTHON
# ===========================================================================

option (BUILD_PYTHON
        "Build the Python bindings"
        ON)

if (BUILD_PYTHON)
    add_subdirectory (python)
endif ()

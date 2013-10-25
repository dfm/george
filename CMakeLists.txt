# George - A C library for blazingly fast Gaussian processes for regression.
# https://github.com/dfm/george
#
# Copyright 2012-2013 Daniel Foreman-Mackey
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# This CMakeLists.txt file is based on the build environment for ceres-solver:
# http://code.google.com/p/ceres-solver

# CMake setup.
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
CMAKE_POLICY(VERSION 2.8)
IF (COMMAND cmake_policy)
    CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

# Start the project.
PROJECT(GEORGE C)

SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Set the output directories.
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# The library version.
SET(GEORGE_VERSION_MAJOR 0)
SET(GEORGE_VERSION_MINOR 2)
SET(GEORGE_VERSION_PATCH 0)
SET(GEORGE_VERSION
    ${GEORGE_VERSION_MAJOR}.${GEORGE_VERSION_MINOR}.${GEORGE_VERSION_PATCH})
SET(GEORGE_ABI_VERSION 0.2.0)
CONFIGURE_FILE (
    "george/george.h.in"
    "${CMAKE_BINARY_DIR}/george/george.h"
)

# Set up testing.
ENABLE_TESTING()

FIND_PACKAGE(LAPACK QUIET)
IF (LAPACK_FOUND)
    MESSAGE("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
ELSE (LAPACK_FOUND)
    MESSAGE(FATAL_ERROR "-- Did not find LAPACK")
ENDIF (LAPACK_FOUND)
FIND_PACKAGE(BLAS QUIET)
IF (BLAS_FOUND)
    MESSAGE("-- Found BLAS library: ${BLAS_LIBRARIES}")
ELSE (BLAS_FOUND)
    MESSAGE(FATAL_ERROR "-- Did not find BLAS library")
ENDIF (BLAS_FOUND)

# Check for SuiteSparse and dependencies.
FIND_PACKAGE(SuiteSparse QUIET)
IF (SUITESPARSE_FOUND)
    # On Ubuntu the system install of SuiteSparse (v3.4.0) up to at least
    # Ubuntu 13.10 cannot be used to link shared libraries.
    IF (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
        MESSAGE(FATAL_ERROR "You are attempting to build a shared "
            "library on Ubuntu using a system package install of SuiteSparse "
            "3.4.0. This package is broken and does not support the "
            "construction of shared libraries.")
    ENDIF (SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

    MESSAGE("-- Found SuiteSparse ${SUITESPARSE_VERSION}")
ELSE (SUITESPARSE_FOUND)
    MESSAGE(FATAL_ERROR "-- Did not find SuiteSparse")
ENDIF (SUITESPARSE_FOUND)

# LIB L-BFGS

FIND_LIBRARY(LBFGS_LIB NAMES lbfgs PATHS ${SEARCH_LIBS})
IF (EXISTS ${LBFGS_LIB})
    MESSAGE("-- Found L-BFGS library: ${LBFGS_LIB}")
ELSE (EXISTS ${LBFGS_LIB})
    MESSAGE(FATAL_ERROR "-- Did not find L-BFGS library")
ENDIF (EXISTS ${LBFGS_LIB})
FIND_PATH(LBFGS_INCLUDE NAMES lbfgs.h PATHS ${SEARCH_HEADERS})
IF (EXISTS ${LBFGS_INCLUDE})
    MESSAGE("-- Found L-BFGS header in: ${LBFGS_INCLUDE}")
ELSE (EXISTS ${LBFGS_INCLUDE})
    MESSAGE(FATAL_ERROR "-- Did not find L-BFGS header")
ENDIF (EXISTS ${LBFGS_INCLUDE})

# ===========================================================================
#   RELEASE BUILD
# ===========================================================================

SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)


# ===========================================================================
#   OPTIMIZATIONS
# ===========================================================================

SET (GEORGE_CXX_FLAGS)
IF (CMAKE_COMPILER_IS_GNUCXX)
    # Linux
    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        SET (GEORGE_CXX_FLAGS "${GEORGE_CXX_FLAGS} -march=native -mtune=native")
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Mac OS X
    IF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        SET (GEORGE_CXX_FLAGS "${GEORGE_CXX_FLAGS} -msse3")
        # Use of -fast only applicable for Apple's GCC
        # Assume this is being used if GCC version < 4.3 on OSX
        EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        IF (GCC_VERSION VERSION_LESS 4.3)
            SET (GEORGE_CXX_FLAGS "${GEORGE_CXX_FLAGS} -fast")
        ENDIF (GCC_VERSION VERSION_LESS 4.3)
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)

IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use of -O4 requires use of gold linker & LLVM-gold plugin, which might
    # well not be present / in use and without which files will compile, but
    # not link ('file not recognized') so explicitly check for support
    INCLUDE(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-O4" HAVE_LTO_SUPPORT)
    IF (HAVE_LTO_SUPPORT)
        MESSAGE(STATUS "Enabling link-time optimization (-O4)")
        SET(GEORGE_CXX_FLAGS "${GEORGE_CXX_FLAGS} -O4")
    ELSE ()
        MESSAGE(STATUS "Compiler/linker does not support link-time optimization (-O4), disabling.")
    ENDIF (HAVE_LTO_SUPPORT)
ENDIF ()

SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GEORGE_CXX_FLAGS}")

IF (UNIX)
    # GCC is not strict enough by default, so enable most of the warnings.
    SET(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
ENDIF (UNIX)

IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600 -Wno-return-type-c-linkage")
ENDIF ()


# ===========================================================================
#   BUILD AND INSTALL
# ===========================================================================

# Heuristic for determining LIB_SUFFIX. FHS recommends that 64-bit systems
# install native libraries to lib64 rather than lib. Most distros seem to
# follow this convention with a couple notable exceptions (Debian-based and
# Arch-based distros) which we try to detect here.
IF (CMAKE_SYSTEM_NAME MATCHES "Linux" AND
        NOT DEFINED LIB_SUFFIX AND
        NOT CMAKE_CROSSCOMPILING AND
        CMAKE_SIZEOF_VOID_P EQUAL "8" AND
        NOT EXISTS "/etc/debian_version" AND
        NOT EXISTS "/etc/arch-release")
    SET(LIB_SUFFIX "64")
ENDIF ()

# Includes.
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/george")
INCLUDE_DIRECTORIES(${SUITESPARSE_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${LBFGS_INCLUDE})

# Deps.
SET(GEORGE_LIBRARY_DEPENDENCIES)
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${SUITESPARSE_LIBRARIES})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${LAPACK_LIBRARIES})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${BLAS_LIBIBRARIES})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${LBFGS_LIB})

# The source file.
SET(GEORGE_SRC george/george.c)

# Install the header.
INSTALL(FILES "${CMAKE_BINARY_DIR}/george/george.h" DESTINATION include)

# Static library.
ADD_LIBRARY(george STATIC ${GEORGE_SRC})
TARGET_LINK_LIBRARIES(george ${GEORGE_LIBRARY_DEPENDENCIES})
INSTALL(TARGETS george
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib${LIB_SUFFIX}
        ARCHIVE DESTINATION lib${LIB_SUFFIX})

# Shared library.
ADD_LIBRARY(george_shared SHARED ${GEORGE_SRC})
TARGET_LINK_LIBRARIES(george_shared ${GEORGE_LIBRARY_DEPENDENCIES})
SET_TARGET_PROPERTIES(george_shared PROPERTIES
                      VERSION ${GEORGE_VERSION}
                      SOVERSION ${GEORGE_ABI_VERSION})

INSTALL(TARGETS george_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib${LIB_SUFFIX}
        ARCHIVE DESTINATION lib${LIB_SUFFIX})


# ===========================================================================
#   TESTING
# ===========================================================================

MACRO (GEORGE_TEST NAME)
    ADD_EXECUTABLE(${NAME}_test george/${NAME}_test.c)
    TARGET_LINK_LIBRARIES(${NAME}_test george)
    ADD_TEST(NAME ${NAME}_test
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test)
ENDMACRO (GEORGE_TEST)

GEORGE_TEST (cholmod_version)
GEORGE_TEST (kernel_gradient)
GEORGE_TEST (hyper_gradient)
GEORGE_TEST (predict)
GEORGE_TEST (optimization)


# ===========================================================================
#   PYTHON
# ===========================================================================

OPTION(BUILD_PYTHON
       "Build the Python bindings"
       ON)

IF (BUILD_PYTHON)
    ADD_SUBDIRECTORY (python)
ENDIF ()

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
find_package (SUITESPARSE REQUIRED)
find_package (LBFGS REQUIRED)

# GCC is not strict enough by default, so enable most of the warnings.
if (UNIX)
    set (CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
endif (UNIX)

# Heuristic for determining LIB_SUFFIX. FHS recommends that 64-bit systems
# install native libraries to lib64 rather than lib. Most distros seem to
# follow this convention with a couple notable exceptions (Debian-based and
# Arch-based distros) which we try to detect here.
if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND
        NOT DEFINED LIB_SUFFIX AND
        NOT CMAKE_CROSSCOMPILING AND
        CMAKE_SIZEOF_VOID_P EQUAL "8" AND
        NOT EXISTS "/etc/debian_version" AND
        NOT EXISTS "/etc/arch-release")
    set (LIB_SUFFIX "64")
endif ()

# Includes.
include_directories ("${CMAKE_BINARY_DIR}/george"
                     ${SUITESPARSE_INCLUDE_DIRS}
                     ${LBFGS_INCLUDE_DIR})

# Deps.
SET(GEORGE_LIBRARY_DEPENDENCIES)
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${CHOLMOD_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${CCOLAMD_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${CAMD_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${COLAMD_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${AMD_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${SUITESPARSE_CONFIG_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${CXSPARSE_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${LAPACK_LIB})
LIST(APPEND GEORGE_LIBRARY_DEPENDENCIES ${BLAS_LIB})
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
#   BENCHMARK
# ===========================================================================

ADD_EXECUTABLE(benchmark george/benchmark.c)
TARGET_LINK_LIBRARIES(benchmark george)


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

configure_file ("george.h.in"
                "${CMAKE_BINARY_DIR}/george/george.h")
install (FILES "${CMAKE_BINARY_DIR}/george/george.h" DESTINATION include)
set (GEORGE_SRC george.c)

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

set (GEORGE_LIBRARY_DEPENDENCIES ${LAPACK_LIBRARIES})
list (APPEND GEORGE_LIBRARY_DEPENDENCIES ${BLAS_LIBRARIES})
list (APPEND GEORGE_LIBRARY_DEPENDENCIES ${SUITESPARSE_LIBRARIES})
if (LBFGS)
    list (APPEND GEORGE_LIBRARY_DEPENDENCIES ${LBFGS_LIBRARY})
endif (LBFGS)

# Build and install the libraries.
add_library (george STATIC ${GEORGE_SRC})
target_link_libraries (george ${GEORGE_LIBRARY_DEPENDENCIES})
install (TARGETS george
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib${LIB_SUFFIX}
         ARCHIVE DESTINATION lib${LIB_SUFFIX})

add_library (george_shared SHARED ${GEORGE_SRC})
target_link_libraries (george_shared ${GEORGE_LIBRARY_DEPENDENCIES})
set_target_properties (george_shared PROPERTIES
                       VERSION ${GEORGE_VERSION}
                       SOVERSION ${GEORGE_ABI_VERSION})
install (TARGETS george_shared
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib${LIB_SUFFIX}
         ARCHIVE DESTINATION lib${LIB_SUFFIX})

# ===========================================================================
#   BENCHMARK AND TESTS
# ===========================================================================

add_executable (benchmark benchmark.c)
target_link_libraries (benchmark george_shared m)

macro (GEORGE_TEST NAME)
    add_executable (${NAME}_test ${NAME}_test.c)
    target_link_libraries (${NAME}_test george_shared m)
    add_test (NAME ${NAME}_test
              COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test)
endmacro (GEORGE_TEST)

george_test (cholmod_version)
george_test (kernel_gradient)
george_test (hyper_gradient)
george_test (predict)
if (LBFGS)
    george_test (optimization)
endif (LBFGS)

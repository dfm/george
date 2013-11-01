# Specify search directories for include files and libraries (this is the union
# of the search directories for all OSs).  Search user-specified hint
# directories first if supplied.
list (APPEND LBFGS_CHECK_INCLUDE_DIRS
      ${LBFGS_INCLUDE_DIR_HINTS}
      /opt/local/include
      /usr/include
      /usr/local/homebrew/include
      /usr/local/include)
list (APPEND LBFGS_CHECK_LIBRARY_DIRS
      ${LBFGS_LIBRARY_DIR_HINTS}
      /opt/local/lib
      /usr/lib
      /usr/local/homebrew/lib
      /usr/local/lib)

set (LBFGS_FOUND TRUE)
find_library (LBFGS_LIBRARY NAMES lbfgs PATHS ${LBFGS_CHECK_LIBRARY_DIRS})
if (EXISTS ${LBFGS_LIBRARY})
    message (STATUS "Found L-BFGS library: ${LBFGS_LIBRARY}")
else (EXISTS ${LBFGS_LIBRARY})
    message ("Did not find L-BFGS library.")
    set (LBFGS_FOUND FALSE)
endif (EXISTS ${LBFGS_LIBRARY})
mark_as_advanced (LBFGS_LIBRARY)

find_path (LBFGS_INCLUDE_DIR NAMES lbfgs.h PATHS ${LBFGS_CHECK_INCLUDE_DIRS})
if (EXISTS ${LBFGS_INCLUDE_DIR})
    message (STATUS "Found L-BFGS header in: ${LBFGS_INCLUDE_DIR}")
else (EXISTS ${LBFGS_INCLUDE_DIR})
    message ("Did not find L-BFGS header.")
    set (LBFGS_FOUND FALSE)
endif (EXISTS ${LBFGS_INCLUDE_DIR})
mark_as_advanced (LBFGS_INCLUDE_DIR)

# Handle REQUIRED and QUIET arguments to FIND_PACKAGE.
include (FindPackageHandleStandardArgs)

# Hack.
set (LBFGS_FOUND_COPY ${LBFGS_FOUND})
find_package_handle_standard_args (LBFGS REQUIRED_VARS LBFGS_FOUND_COPY
                                   LBFGS_INCLUDE_DIR LBFGS_LIBRARY)

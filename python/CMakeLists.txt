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
# This CMakeLists.txt file is based on:
# http://bloerg.net/2012/11/10/cmake-and-distutils.html

FIND_PROGRAM (PYTHON "python")

if (PYTHON)
    MESSAGE("-- Found Python interpreter: ${PYTHON}")

    SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    SET(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    SET(INIT_PY_IN  "${CMAKE_CURRENT_SOURCE_DIR}/george/__init__.py.in")
    SET(INIT_PY     "${CMAKE_CURRENT_SOURCE_DIR}/george/__init__.py")
    SET(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/george/*.py")
    SET(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    CONFIGURE_FILE(${SETUP_PY_IN} ${SETUP_PY})
    CONFIGURE_FILE(${INIT_PY_IN} ${INIT_PY})

    ADD_CUSTOM_COMMAND(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    ADD_CUSTOM_TARGET(python ALL DEPENDS ${OUTPUT})

    INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${PYTHON} ${SETUP_PY} install)")

ELSE (PYTHON)
    MESSAGE("-- Did not find Python interpreter")
ENDIF()

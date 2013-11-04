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

find_program (PYTHON "python")

if (PYTHON)
    message ("-- Found Python interpreter: ${PYTHON}")

    set (SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set (SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set (INIT_PY_IN  "${CMAKE_CURRENT_SOURCE_DIR}/george/__init__.py.in")
    set (INIT_PY     "${CMAKE_CURRENT_SOURCE_DIR}/george/__init__.py")
    set (DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/george/*.py")
    set (OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file (${SETUP_PY_IN} ${SETUP_PY})
    configure_file (${INIT_PY_IN} ${INIT_PY})

    add_custom_command (OUTPUT ${OUTPUT}
                        COMMAND ${PYTHON} ${SETUP_PY} build
                        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                        DEPENDS ${DEPS})

    add_custom_target (python ALL DEPENDS ${OUTPUT})

    install (CODE
             "EXECUTE_PROCESS(COMMAND ${PYTHON} ${SETUP_PY} install ${PYTHON_INSTALL_OPTIONS})")

else (PYTHON)
    message ("-- Did not find Python interpreter")
endif ()

# distutils: language = c++
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "solver.h" namespace "george":

    cdef cppclass HODLRSolver:
        pass

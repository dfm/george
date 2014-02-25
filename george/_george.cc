#include <Python.h>
#include <numpy/arrayobject.h>
#include <Eigen/Dense>

#include "george.h"

using george::HODLRSolver;
using george::Kernel;

using george::SumKernel;
using george::ProductKernel;

using george::ExpKernel;
using george::SparseKernel;
using george::CosineKernel;
using george::ExpSquaredKernel;

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_IN_ARRAY)

extern "C" {

// Type definition.
typedef struct {

    PyObject_HEAD
    Kernel* kernel;
    HODLRSolver<Kernel>* solver;

} _george_object;

// Allocation/deallocation.
static void _george_dealloc (_george_object* self);
static PyObject* _george_new (PyTypeObject* type, PyObject* args, PyObject* kwds);
static int _george_init (_george_object* self, PyObject* args, PyObject* kwds);

// Methods.
static PyObject* _george_compute (_george_object* self, PyObject* args);
static PyObject* _george_lnlikelihood (_george_object* self, PyObject* args);
static PyObject* _george_computed (_george_object* self, PyObject* args);
static PyObject* _george_predict (_george_object* self, PyObject* args);
static PyObject* _george_get_matrix (_george_object* self, PyObject* args);

// Module interactions.
void init_george (void);

}

static void _george_dealloc (_george_object* self)
{
    if (self->kernel != NULL) delete self->kernel;
    if (self->solver != NULL) delete self->solver;
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *_george_new (PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    _george_object* self = (_george_object*)type->tp_alloc(type, 0);
    self->kernel = NULL;
    self->solver = NULL;
    return (PyObject*)self;
}

Kernel* parse_kernel (PyObject* kernel)
{
    // Check the kernel.
    if (!PyObject_HasAttrString(kernel, "is_kernel")) {
        PyErr_SetString(PyExc_TypeError, "Invalid kernel");
        return NULL;
    }

    // Deal with operators.
    PyObject* is_kernel_obj = PyObject_GetAttrString(kernel, "is_kernel");
    if (is_kernel_obj == Py_False) {
        Py_DECREF(is_kernel_obj);

        // Get the Python kernel objects and the type of operator.
        PyObject* k1 = PyObject_GetAttrString(kernel, "k1"),
                * k2 = PyObject_GetAttrString(kernel, "k2"),
                * otype_obj = PyObject_GetAttrString(kernel, "operator_type");
        long otype = PyInt_AsLong(otype_obj);
        Py_XDECREF(otype_obj);

        // Validate the kernels.
        if (k1 == NULL || k2 == NULL || PyErr_Occurred() != NULL) {
            PyErr_SetString(PyExc_TypeError, "Invalid operator");
            Py_XDECREF(k1);
            Py_XDECREF(k2);
            return NULL;
        }

        // Parse the combined kernels.
        Kernel* kernel1 = parse_kernel(k1);
        Kernel* kernel2 = parse_kernel(k2);
        Py_DECREF(k1);
        Py_DECREF(k2);

        // Check that these were successfully parsed.
        if (kernel1 == NULL || kernel2 == NULL) {
            if (kernel1 != NULL) delete kernel1;
            if (kernel2 != NULL) delete kernel2;
            return NULL;
        }

        // Combine the kernels.
        if (otype == 0)
            return new SumKernel (kernel1, kernel2);
        else if (otype == 1)
            return new ProductKernel (kernel1, kernel2);

        // If we get here then the operator type was unknown.
        if (kernel1 != NULL) delete kernel1;
        if (kernel2 != NULL) delete kernel2;
        PyErr_SetString(PyExc_TypeError, "Unknown operator");
        return NULL;
    }

    // If we get here then the object claims to be a kernel.
    Py_DECREF(is_kernel_obj);

    // Get the kernel type.
    PyObject* ktype_obj = PyObject_GetAttrString(kernel, "kernel_type");
    long ktype = PyInt_AsLong(ktype_obj);
    Py_XDECREF(ktype_obj);
    if (ktype_obj == NULL || PyErr_Occurred() != NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid kernel type");
        return NULL;
    }

    // Get the parameter vector.
    PyObject* pars_obj = PyObject_GetAttrString(kernel, "pars");
    PyArrayObject* pars_array = PARSE_ARRAY(pars_obj);
    Py_XDECREF(pars_obj);
    if (pars_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid parameter vector");
        Py_XDECREF(pars_array);
        return NULL;
    }

    // Get the number of parameters and a pointer to the parameter data.
    int npars = PyArray_DIM(pars_array, 0);
    double* pars = (double*)PyArray_DATA(pars_array);

    // Build the kernel.
    Kernel* k = NULL;
    if (ktype == 0) {
        if (npars != 2) {
            PyErr_SetString(PyExc_ValueError,
                    "The ExpSquaredKernel requires 2 parameters.");
            Py_DECREF(pars_array);
            return NULL;
        }
        k = new ExpSquaredKernel(pars[0], pars[1]);
    } else if (ktype == 1) {
        if (npars != 2) {
            PyErr_SetString(PyExc_ValueError,
                    "The ExpKernel requires 2 parameters.");
            Py_DECREF(pars_array);
            return NULL;
        }
        k = new ExpKernel(pars[0], pars[1]);
    } else if (ktype == 2) {
        if (npars != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "The CosineKernel requires 1 parameter.");
            Py_DECREF(pars_array);
            return NULL;
        }
        k = new CosineKernel(pars[0]);
    } else if (ktype == 3) {
        if (npars != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "The SparseKernel requires 1 parameter.");
            Py_DECREF(pars_array);
            return NULL;
        }
        k = new SparseKernel(pars[0]);
    } else PyErr_SetString(PyExc_TypeError, "Unknown kernel");

    Py_DECREF(pars_array);
    return k;
}

static int _george_init(_george_object* self, PyObject* args, PyObject* kwds)
{
    int nleaf;
    double tol;
    PyObject* kernel = NULL;
    if (!PyArg_ParseTuple(args, "Oid", &kernel, &nleaf, &tol))
        return -1;

    // Parse the kernel.
    self->kernel = parse_kernel (kernel);
    if (self->kernel == NULL) return -2;

    // // Parse the parameter vector.
    // PyArrayObject* pars_array = PARSE_ARRAY(pars_obj);
    // if (pars_array == NULL) return -1;

    // // Get and check the number of parameters.
    // int npars = PyArray_DIM(pars_array, 0);

    // if (ktype == 0) {
    //     if (npars != 2) {
    //         PyErr_SetString(PyExc_RuntimeError,
    //             "The kernel takes exactly 2 parameters.");
    //         Py_DECREF(pars_array);
    //         return -2;
    //     }

    //     // Set up the kernel.
    //     double* pars = (double*)PyArray_DATA(pars_array);
    //     self->kernel = new ExpSquaredKernel (pars);
    // } else if (ktype == 1) {
    //     if (npars != 4) {
    //         PyErr_SetString(PyExc_RuntimeError,
    //             "The kernel takes exactly 4 parameters.");
    //         Py_DECREF(pars_array);
    //         return -2;
    //     }

    //     // Set up the kernel.
    //     double* pars = (double*)PyArray_DATA(pars_array);
    //     ExpSquaredKernel* k1 = new ExpSquaredKernel (pars);
    //     ExpSquaredKernel* k2 = new ExpSquaredKernel (&(pars[2]));
    //     self->kernel = new MixtureKernel<ExpSquaredKernel, ExpSquaredKernel> (k1, k2);

    // } else if (ktype == 2) {
    //     if (npars != 3) {
    //         PyErr_SetString(PyExc_RuntimeError,
    //             "The kernel takes exactly 3 parameters.");
    //         Py_DECREF(pars_array);
    //         return -2;
    //     }

    //     // Set up the kernel.
    //     double* pars = (double*)PyArray_DATA(pars_array);
    //     ExpKernel* k1 = new ExpKernel (pars);
    //     CosineKernel* k2 = new CosineKernel (&(pars[2]));
    //     self->kernel = new ProductKernel<ExpKernel, CosineKernel> (k1, k2);

    // } else if (ktype == 3) {
    //     if (npars != 5) {
    //         PyErr_SetString(PyExc_RuntimeError,
    //             "The kernel takes exactly 5 parameters.");
    //         Py_DECREF(pars_array);
    //         return -2;
    //     }

    //     // Set up the kernel.
    //     double* pars = (double*)PyArray_DATA(pars_array);
    //     ExpSquaredKernel* k1 = new ExpSquaredKernel (pars);
    //     ExpKernel* k2 = new ExpKernel (&(pars[2]));
    //     CosineKernel* k3 = new CosineKernel (&(pars[4]));
    //     self->kernel =
    //         new MixtureKernel<ExpSquaredKernel,
    //                           ProductKernel<ExpKernel, CosineKernel> >
    //             (k1, new ProductKernel<ExpKernel, CosineKernel> (k2, k3));
    // } else {
    //     PyErr_SetString(PyExc_RuntimeError, "Unknown kernel type");
    //     Py_DECREF(pars_array);
    //     return -3;
    // }
    // Py_DECREF(pars_array);

    // Set up the solver.
    self->solver = new HODLRSolver<Kernel> (self->kernel, nleaf, tol);

    return 0;
}

static PyObject* _george_compute (_george_object* self, PyObject* args)
{
    PyObject* x_obj, * yerr_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &yerr_obj))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject * x_array = PARSE_ARRAY(x_obj),
                  * yerr_array = PARSE_ARRAY(yerr_obj);
    if (x_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(yerr_array);
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse input objects as numpy arrays");
        return NULL;
    }

    // Get the dimensions.
    int nsamples = (int)PyArray_DIM(x_array, 0);
    if ((int)PyArray_NDIM(x_array) >= 2) {
        Py_DECREF(x_array);
        Py_DECREF(yerr_array);
        PyErr_SetString(PyExc_ValueError, "George only works in 1D for now.");
        return NULL;
    }
    if ((int)PyArray_DIM(yerr_array, 0) != nsamples) {
        Py_DECREF(x_array);
        Py_DECREF(yerr_array);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        return NULL;
    }

    // Access the data.
    double * x = (double*)PyArray_DATA(x_array),
           * yerr = (double*)PyArray_DATA(yerr_array);

    // Map to vectors.
    VectorXd x_vec = VectorXd::Map(x, nsamples),
             yerr_vec = VectorXd::Map(yerr, nsamples);

    // Pre-compute the factorization.
    int info = self->solver->compute (x_vec, yerr_vec);

    // Clean up.
    Py_DECREF(x_array);
    Py_DECREF(yerr_array);

    // Check success.
    if (info != george::SOLVER_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute model");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* _george_lnlikelihood (_george_object* self, PyObject* args)
{
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;

    if (!self->solver->get_computed()) {
        PyErr_SetString(PyExc_RuntimeError,
            "You need to compute the model first");
        return NULL;
    }

    PyArrayObject *y_array = PARSE_ARRAY(y_obj);
    if (y_array == NULL) {
        Py_XDECREF(y_array);
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse input object as a numpy array");
        return NULL;
    }

    // Get the data.
    int nsamples = (int)PyArray_DIM(y_array, 0);
    double* y = (double*)PyArray_DATA(y_array);
    VectorXd y_vec = VectorXd::Map(y, nsamples);

    // Compute the likelihood.
    double lnlike = self->solver->log_likelihood(y_vec);
    Py_DECREF(y_array);

    // Check success.
    if (self->solver->get_status() != george::SOLVER_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute likelihood");
        return NULL;
    }

    return Py_BuildValue("d", lnlike);
}

static PyObject* _george_predict (_george_object* self, PyObject* args)
{
    PyObject* y_obj, * x_obj;
    if (!PyArg_ParseTuple(args, "OO", &y_obj, &x_obj)) return NULL;

    if (!self->solver->get_computed()) {
        PyErr_SetString(PyExc_RuntimeError,
            "You need to compute the model first");
        return NULL;
    }

    PyArrayObject *y_array = PARSE_ARRAY(y_obj),
                  *x_array = PARSE_ARRAY(x_obj);
    if (y_array == NULL || x_array == NULL) {
        Py_XDECREF(y_array);
        Py_XDECREF(x_array);
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse input objects as numpy arrays");
        return NULL;
    }

    // Get the dimensions.
    int nsamples = (int)PyArray_DIM(y_array, 0),
        ntest = (int)PyArray_DIM(x_array, 0);
    if ((int)PyArray_NDIM(x_array) >= 2) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "George only works in 1D for now.");
        return NULL;
    }
    if (self->solver->get_dimension() != nsamples) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        return NULL;
    }

    // Access the data.
    double* y = (double*) PyArray_DATA(y_array),
          * x = (double*) PyArray_DATA(x_array);

    // Compute the mean.
    VectorXd y_vec = VectorXd::Map(y, nsamples),
             x_vec = VectorXd::Map(x, ntest),
             mu_vec;
    MatrixXd cov_mat;
    self->solver->predict(y_vec, x_vec, mu_vec, cov_mat);

    // Clean up.
    Py_DECREF(y_array);
    Py_DECREF(x_array);

    // Check success.
    if (self->solver->get_status() != george::SOLVER_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute model");
        return NULL;
    }

    // Allocate the output arrays.
    npy_intp dim[] = {ntest}, dim2[] = {ntest, ntest};
    PyArrayObject* mu_array = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE),
                 * cov_array = (PyArrayObject*)PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
    if (mu_array == NULL || cov_array == NULL) {
        Py_XDECREF(mu_array);
        Py_XDECREF(cov_array);
        return NULL;
    }

    // Copy over the result.
    double *mu = (double*)PyArray_DATA(mu_array),
           *cov = (double*)PyArray_DATA(cov_array);
    for (int i = 0; i < ntest; ++i) {
        mu[i] = mu_vec[i];
        for (int j = 0; j < ntest; ++j) cov[i*ntest+j] = cov_mat(i, j);
    }

    // Build the result.
    PyObject *ret = Py_BuildValue("OO", mu_array, cov_array);
    Py_DECREF(mu_array);
    Py_DECREF(cov_array);

    if (ret == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output tuple");
        Py_XDECREF(ret);
        return NULL;
    }

    return ret;
}

static PyObject* _george_get_matrix (_george_object* self, PyObject* args)
{
    PyObject* t_obj;
    if (!PyArg_ParseTuple(args, "O", &t_obj)) return NULL;

    PyArrayObject* t_array = PARSE_ARRAY(t_obj);
    if (t_array == NULL) {
        Py_XDECREF(t_array);
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse input object as numpy array");
        return NULL;
    }

    // Get the dimensions.
    int n = (int)PyArray_DIM(t_array, 0);
    if ((int)PyArray_NDIM(t_array) >= 2) {
        Py_DECREF(t_array);
        PyErr_SetString(PyExc_ValueError, "George only works in 1D for now.");
        return NULL;
    }

    // Access the data.
    double* t = (double*) PyArray_DATA(t_array);

    // Allocate the output arrays.
    npy_intp dim[] = {n, n};
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(2, dim, NPY_DOUBLE);
    if (out_array == NULL) {
        Py_DECREF(t_array);
        Py_XDECREF(out_array);
        return NULL;
    }

    // Copy over the result.
    int flag;
    double value, *matrix = (double*)PyArray_DATA(out_array);
    Kernel* kernel = self->kernel;
    for (int i = 0; i < n; ++i) {
        matrix[i*n+i] = kernel->evaluate(t[i], t[i], &flag);
        for (int j = i+1; j < n; ++j) {
            value = kernel->evaluate(t[i], t[j], &flag);
            matrix[i*n+j] = value;
            matrix[j*n+i] = value;
        }
    }

    Py_DECREF(t_array);
    return (PyObject*)out_array;
}

static PyObject* _george_computed(_george_object* self, PyObject* args)
{
    if (self->solver->get_computed()) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef _george_methods[] = {
    {"compute",
     (PyCFunction)_george_compute,
     METH_VARARGS,
     "Fit the GP."},
    {"lnlikelihood",
     (PyCFunction)_george_lnlikelihood,
     METH_VARARGS,
     "Get the marginalized ln likelihood of some values."
    },
    {"predict",
     (PyCFunction)_george_predict,
     METH_VARARGS,
     "Predict the mean function"
    },
    {"get_matrix",
     (PyCFunction)_george_get_matrix,
     METH_VARARGS,
     "Get the covariance matrix for a set of times."
    },
    {"computed",
     (PyCFunction)_george_computed,
     METH_NOARGS,
     "Has the GP been computed?"
    },
    {NULL}  /* Sentinel */
};

static char _george_doc[] = "This is the ``_george`` object. "
                            "There is some black magic.";
static PyTypeObject _george_type = {
    PyObject_HEAD_INIT(NULL)
    0,                           /*ob_size*/
    "_george._george",           /*tp_name*/
    sizeof(_george_object),      /*tp_basicsize*/
    0,                           /*tp_itemsize*/
    (destructor)_george_dealloc, /*tp_dealloc*/
    0,                           /*tp_print*/
    0,                           /*tp_getattr*/
    0,                           /*tp_setattr*/
    0,                           /*tp_compare*/
    0,                           /*tp_repr*/
    0,                           /*tp_as_number*/
    0,                           /*tp_as_sequence*/
    0,                           /*tp_as_mapping*/
    0,                           /*tp_hash */
    0,                           /*tp_call*/
    0,                           /*tp_str*/
    0,                           /*tp_getattro*/
    0,                           /*tp_setattro*/
    0,                           /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                 /*tp_flags*/
    _george_doc,                 /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    _george_methods,             /* tp_methods */
    0,                           /* tp_members */
    0,                           /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    (initproc)_george_init,      /* tp_init */
    0,                           /* tp_alloc */
    _george_new,                 /* tp_new */
};


//
// Initialize the module.
//

static char module_doc[] = "GP Module";
static PyMethodDef module_methods[] = {{NULL}};
void init_george(void)
{
    PyObject *m;

    if (PyType_Ready(&_george_type) < 0)
        return;

    m = Py_InitModule3("_george", module_methods, module_doc);
    if (m == NULL)
        return;

    Py_INCREF(&_george_type);
    PyModule_AddObject(m, "_george", (PyObject *)&_george_type);

    import_array();
}

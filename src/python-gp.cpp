#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "kernels.h"
#include "gp.h"


using namespace Eigen;


//
// Shortcut function for parsing input numpy arrays and vectors.
//

template <class T> int nparray2matrix(PyObject *obj, T *m)
{
    // Load the array.
    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (array == NULL) {
        PyErr_SetString(PyExc_TypeError, "The input object is not an array.");
        return 1;
    }

    // Get the dimensions.
    int ndim = PyArray_NDIM(array);
    if (ndim == 0 || ndim > 2) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_TypeError,
                            "The input array can only be 1 or 2 dimensional");
        return 2;
    }

    // Get the shape.
    int N = PyArray_DIM(array, 0), M = 1;
    if (ndim == 2)
        M = PyArray_DIM(array, 1);
    (*m).resize(N, M);

    // Access the data.
    double *data = (double*)PyArray_DATA(array);
    int i, j;
    for (i = 0; i < N; ++i)
        for (j = 0; j < M; ++j)
            (*m)(i, j) = data[i * M + j];

    Py_DECREF(array);
    return 0;
}


//
// The ``_gp`` type definition.
//

typedef struct {
    PyObject_HEAD
    GaussianProcess gp;
} _gp;

static void _gp_dealloc(_gp *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *_gp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    _gp *self;

    self = (_gp*)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int _gp_init(_gp *self, PyObject *args, PyObject *kwds)
{
    PyObject *parsobj = NULL;
    int kerneltype;

    if (!PyArg_ParseTuple(args, "Oi", &parsobj, &kerneltype))
        return -1;

    VectorXd pars;
    nparray2matrix<VectorXd>(parsobj, &pars);
    if (PyErr_Occurred() != NULL)
        return -1;

    // Which type of kernel?
    double (*k)(VectorXd, VectorXd, VectorXd);
    if (kerneltype == 1) {
        if (pars.rows() != 2) {
            PyErr_SetString(PyExc_RuntimeError,
                            "The isotropic kernel requires 2 parameters.");
            return -1;
        }

        k = isotropicKernel;
    } else if (kerneltype == 2) {
        k = diagonalKernel;
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Unknown kernel type.");
        return -1;
    }

    self->gp = GaussianProcess(pars, k);

    return 0;
}

static PyMemberDef _gp_members[] = {{NULL}};  // We don't haz no members.

static PyObject *_gp_fit(_gp *self, PyObject *args)
{
    PyObject *xobj, *yobj, *yerrobj;
    if (!PyArg_ParseTuple(args, "OOO", &xobj, &yobj, &yerrobj))
        return NULL;

    // Parse the input numpy arrays and cast them as Eigen objects.
    MatrixXd x;
    VectorXd y, yerr;
    nparray2matrix<MatrixXd>(xobj, &x);
    nparray2matrix<VectorXd>(yobj, &y);
    nparray2matrix<VectorXd>(yerrobj, &yerr);
    if (PyErr_Occurred() != NULL)
        return NULL;

    //
    // TODO: Check dimensions + number of parameters against kernel type.
    //

    // Do the fit.
    int ret = self->gp.fit(x, y, yerr);
    if (ret == 1 || ret == 2) {
        PyErr_SetString(PyExc_TypeError, "Dimension mismatch.");
        return NULL;
    } if (ret == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't factorize K.");
        return NULL;
    } else if (ret == -2) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't solve for alpha.");
        return NULL;
    } else if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to evaluate GP.");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_gp_predict(_gp *self, PyObject *args)
{
    PyObject *xobj;
    if (!PyArg_ParseTuple(args, "O", &xobj))
        return NULL;

    // Parse the input numpy arrays and cast them as Eigen objects.
    MatrixXd x;
    nparray2matrix<MatrixXd>(xobj, &x);
    if (PyErr_Occurred() != NULL)
        return NULL;

    // Do the prediction.
    VectorXd mu(1);
    MatrixXd cov(1, 1);
    int ret = self->gp.predict(x, &mu, &cov);
    if (ret == -1) {
        PyErr_SetString(PyExc_RuntimeError, "You have to fit the GP first.");
        return NULL;
    } else if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't solve for the mean/covariance.");
        return NULL;
    }

    // Build the empty numpy arrays for output.
    int n = mu.rows();
    npy_intp dim[] = {n};
    PyObject *mu_array = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);

    npy_intp dims[] = {n, n};
    PyObject *cov_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);

    // Fill in the output data.
    double *mu_data = (double*)PyArray_DATA(mu_array);
    double *cov_data = (double*)PyArray_DATA(cov_array);
    for (int i = 0; i < n; ++i) {
        mu_data[i] = mu[i];
        for (int j = 0; j < n; ++j)
            cov_data[i * n + j] = cov(i, j);
    }

    // Build the full output tuple.
    PyObject *tuple_out = Py_BuildValue("NN", mu_array, cov_array);
    if (tuple_out == NULL) {
        Py_DECREF(mu_array);
        Py_DECREF(cov_array);
        return NULL;
    }

    return tuple_out;
}

static PyObject *_gp_evaluate(_gp *self)
{
    double lnlike = self->gp.evaluate();
    PyObject *tuple_out = Py_BuildValue("d", lnlike);
    return tuple_out;
}

static PyMethodDef _gp_methods[] = {
    {"fit", (PyCFunction)_gp_fit, METH_VARARGS, "Fit the GP."},
    {"evaluate", (PyCFunction)_gp_evaluate, METH_NOARGS, "Evaluate the GP."},
    {"predict", (PyCFunction)_gp_predict, METH_VARARGS, "Predict new values."},
    {NULL}  /* Sentinel */
};

static char _gp_doc[] = "This is the ``_gp`` object. There is some black magic.";
static PyTypeObject _gp_type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_gp._gp",                 /*tp_name*/
    sizeof(_gp),               /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)_gp_dealloc,   /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    _gp_doc,                   /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    _gp_methods,               /* tp_methods */
    _gp_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)_gp_init,        /* tp_init */
    0,                         /* tp_alloc */
    _gp_new,                   /* tp_new */
};


//
// Initialize the module.
//

static char module_doc[] = "GP Module";
static PyMethodDef module_methods[] = {{NULL}};
extern "C" void init_gp(void)
{
    PyObject *m;

    if (PyType_Ready(&_gp_type) < 0)
        return;

    m = Py_InitModule3("_gp", module_methods, module_doc);
    if (m == NULL)
        return;

    Py_INCREF(&_gp_type);
    PyModule_AddObject(m, "_gp", (PyObject *)&_gp_type);

    import_array();
}

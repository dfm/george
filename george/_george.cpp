#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "george.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using George::GaussianProcess;
using George::IsotropicGaussianKernel;

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_IN_ARRAY)

//
// The ``_george`` type definition.
//

typedef struct {
    PyObject_HEAD
    GaussianProcess<IsotropicGaussianKernel> *gp;
} _george;

static void _george_dealloc(_george *self)
{
    delete self->gp;
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *_george_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    _george *self;
    self = (_george*)type->tp_alloc(type, 0);
    self->gp = new GaussianProcess<IsotropicGaussianKernel> ();
    return (PyObject*)self;
}

static int _george_init(_george *self, PyObject *args, PyObject *kwds)
{
    PyObject *pars_obj = NULL;

    if (!PyArg_ParseTuple(args, "O", &pars_obj))
        return -1;

    // Parse the parameter vector.
    PyArrayObject *pars_array = PARSE_ARRAY(pars_obj);
    if (pars_array == NULL) return -1;

    int npars = PyArray_DIM(pars_array, 0);
    double *pars = (double*)PyArray_DATA(pars_array);

    // Which type of kernel?
    if (npars != 3) {
        PyErr_SetString(PyExc_RuntimeError,
            "The isotropic kernel takes exactly 3 parameters.");
        return -1;
    }

    IsotropicGaussianKernel kernel(VectorXd::Map(pars, npars));
    self->gp->set_kernel(kernel);
    return 0;
}

static PyMemberDef _george_members[] = {{NULL}};

static PyObject *_george_compute (_george *self, PyObject *args)
{
    PyObject *x_obj, *yerr_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &yerr_obj))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject *x_array = PARSE_ARRAY(x_obj),
                  *yerr_array = PARSE_ARRAY(yerr_obj);
    if (x_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(yerr_array);
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse input objects as numpy arrays");
        return NULL;
    }

    // Get the dimensions.
    int nsamples = (int)PyArray_DIM(x_array, 0),
        ndim = 1;
    if ((int)PyArray_NDIM(x_array) == 2)
        ndim = (int)PyArray_DIM(x_array, 1);

    if ((int)PyArray_DIM(yerr_array, 0) != nsamples) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        Py_DECREF(x_array);
        Py_DECREF(yerr_array);
        return NULL;
    }

    // Access the data.
    double *x = (double*)PyArray_DATA(x_array),
           *yerr = (double*)PyArray_DATA(yerr_array);

    // Fit the GP.
    int info = self->gp->compute(MatrixXd::Map(x, nsamples, ndim),
                                 VectorXd::Map(yerr, nsamples));

    // Clean up.
    Py_DECREF(x_array);
    Py_DECREF(yerr_array);

    // Check success.
    if (info != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute model");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_george_lnlikelihood(_george *self, PyObject *args)
{
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;

    if (!self->gp->computed()) {
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

    // Get the dimension.
    int nsamples = (int)PyArray_DIM(y_array, 0);
    if (nsamples != self->gp->nsamples()) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        Py_DECREF(y_array);
        return NULL;
    }

    double *y = (double*)PyArray_DATA(y_array),
           lnlike = self->gp->lnlikelihood(VectorXd::Map(y, nsamples));

    // Clean up.
    Py_DECREF(y_array);

    return Py_BuildValue("d", lnlike);
}

static PyObject *_george_predict(_george *self, PyObject *args)
{
    PyObject *y_obj, *x_obj;
    if (!PyArg_ParseTuple(args, "OO", &y_obj, &x_obj)) return NULL;

    if (!self->gp->computed()) {
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

    // Get the dimension.
    int nsamples = (int)PyArray_DIM(y_array, 0),
        ntest = (int)PyArray_DIM(x_array, 0),
        ndim = 1;

    if ((int)PyArray_NDIM(x_array) == 2)
        ndim = (int)PyArray_DIM(x_array, 1);

    if (nsamples != self->gp->nsamples()) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        Py_DECREF(y_array);
        Py_DECREF(x_array);
        return NULL;
    }

    double *y = (double*)PyArray_DATA(y_array),
           *x = (double*)PyArray_DATA(x_array);

    VectorXd mu_vec(ntest);
    MatrixXd cov_vec(ntest, ntest);
    int info = self->gp->predict(VectorXd::Map(y, nsamples),
                                MatrixXd::Map(x, ntest, ndim),
                                &mu_vec, &cov_vec);

    // Clean up.
    Py_DECREF(y_array);
    Py_DECREF(x_array);

    // Check success.
    if (info != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute model");
        return NULL;
    }

    // Allocate the output arrays.
    npy_intp dim[1] = {ntest}, dim2[2] = {ntest, ntest};
    PyArrayObject *mu_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                NPY_DOUBLE),
                  *cov_array = (PyArrayObject*)PyArray_SimpleNew(2, dim2,
                                                                 NPY_DOUBLE);
    if (mu_array == NULL || cov_array == NULL) {
        Py_XDECREF(mu_array);
        Py_XDECREF(cov_array);
        return NULL;
    }

    // Copy the data over.
    double *mu = (double*)PyArray_DATA(mu_array),
           *cov = (double*)PyArray_DATA(cov_array);
    int i, j;
    for (i = 0; i < ntest; ++i) {
        mu[i] = mu_vec(i);
        for (j = 0; j < ntest; ++j)
            cov[i * ntest + j] = cov_vec(i, j);
    }

    return Py_BuildValue("OO", mu_array, cov_array);
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
     "Predict."
    },
    {NULL}  /* Sentinel */
};

static char _george_doc[] = "This is the ``_george`` object. There is some black magic.";
static PyTypeObject _george_type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_george._george",         /*tp_name*/
    sizeof(_george),           /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)_george_dealloc, /*tp_dealloc*/
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
    _george_doc,                   /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    _george_methods,               /* tp_methods */
    _george_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)_george_init,        /* tp_init */
    0,                         /* tp_alloc */
    _george_new,                   /* tp_new */
};


//
// Initialize the module.
//

static char module_doc[] = "GP Module";
static PyMethodDef module_methods[] = {{NULL}};
extern "C" void init_george(void)
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

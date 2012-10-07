#include <Python.h>
#include <numpy/arrayobject.h>
#include "gp.h"


using namespace Eigen;


#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif


static char module_doc[] = "GP Module";
static char evaluate_doc[] = "Evaluate a GP.";


static PyObject *gp_evaluate(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = {
    {"evaluate", gp_evaluate, METH_VARARGS, evaluate_doc},
    {NULL, NULL, 0, NULL}
};


#ifdef IS_PY3K
#define INITERROR return NULL

static struct PyModuleDef moduledef =
            {PyModuleDef_HEAD_INIT, "_gp", module_doc, -1, module_methods, };

extern "C" PyObject *PyInit__gp(void)
{
    PyObject *m = PyModule_Create(&moduledef);

#else
#define INITERROR return

extern "C" void init_gp(void)
{
    PyObject *m = Py_InitModule3("_gp", module_methods, module_doc);
#endif

    if (m == NULL)
        INITERROR;

    import_array();

#ifdef IS_PY3K
    return m;
#endif
}

template <class T>
int nparray2matrix(PyObject *obj, T *m)
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


static PyObject *gp_evaluate(PyObject *self, PyObject *args)
{
    PyObject *xobj, *yobj, *yerrobj, *x0obj, *parsobj;
    int kerneltype;
    if (!PyArg_ParseTuple(args, "OOOOOi", &xobj, &yobj, &yerrobj, &x0obj,
                &parsobj, &kerneltype))
        return NULL;

    // Parse the input numpy arrays and cast them as Eigen objects.
    MatrixXd x, x0;
    VectorXd y, yerr, pars;
    nparray2matrix<MatrixXd>(xobj, &x);
    nparray2matrix<VectorXd>(yobj, &y);
    nparray2matrix<VectorXd>(yerrobj, &yerr);
    nparray2matrix<MatrixXd>(x0obj, &x0);
    nparray2matrix<VectorXd>(parsobj, &pars);

    if (PyErr_Occurred() != NULL)
        return NULL;

    // Which type of kernel?
    double (*k)(VectorXd, VectorXd, VectorXd);
    if (kerneltype == 1) {
        if (pars.rows() != 2) {
            PyErr_SetString(PyExc_RuntimeError,
                            "The isotropic kernel requires 2 parameters.");
            return NULL;
        }

        k = isotropicKernel;
    } else if (kerneltype == 2) {
        if (pars.rows() != 1 + x.cols()) {
            PyErr_SetString(PyExc_RuntimeError,
                        "The diagonal kernel requires 1 + ndim parameters.");
            return NULL;
        }

        k = diagonalKernel;
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Unknown kernel type.");
        return NULL;
    }

    // Run the inference.
    double lnlike = 0.5;
    VectorXd mean(1), variance(1);
    int ret = evaluateGP(x, y, yerr, x0, pars, k, &mean,
                         &variance, &lnlike, 1e-5);

    if (ret == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't factorize K.");
        return NULL;
    } else if (ret == -2) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't solve for alpha.");
        return NULL;
    } else if (ret == -3) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't solve for variance.");
        return NULL;
    } else if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to evaluate GP.");
        return NULL;
    }

    // Build the output objects:
    npy_intp dims[] = {mean.rows()};

    // Build the empty numpy arrays.
    PyObject *mean_array = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    PyObject *variance_array = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);

    // Fill the data pointer.
    double *mean_data = (double*)PyArray_DATA(mean_array);
    double *variance_data = (double*)PyArray_DATA(variance_array);
    for (int i = 0; i < mean.rows(); ++i) {
        mean_data[i] = mean[i];
        variance_data[i] = variance[i];
    }

    // Build the full output tuple.
    PyObject *tuple_out = Py_BuildValue("NNd", mean_array, variance_array,
                                        lnlike);
    if (tuple_out == NULL) {
        Py_DECREF(mean_array);
        Py_DECREF(variance_array);
        return NULL;
    }

    return tuple_out;
}

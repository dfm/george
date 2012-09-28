#include <Python.h>
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

#ifdef IS_PY3K
    return m;
#endif
}


static PyObject *gp_evaluate(PyObject *self, PyObject *args)
{
    PyObject *xobj, *yobj, *yerrobj, *x0obj;
    if (!PyArg_ParseTuple(args, "OOOO", &xobj, &yobj, &yerrobj, &x0obj))
        return NULL;

    /* Read in the data and convert to Eigen::VectorXd objects */
    Py_ssize_t n = PyList_Size(xobj);
    if (PyErr_Occurred() != NULL) {
        PyErr_SetString(PyExc_TypeError, "The data objects must be lists.");
        return NULL;
    }
    if (n != PyList_Size(yobj) || n != PyList_Size(yerrobj)) {
        PyErr_SetString(PyExc_TypeError, "Dimension mismatch.");
        return NULL;
    }
    VectorXd x(n), y(n), yerr(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        x[i] = PyFloat_AsDouble(PyList_GetItem(xobj, i));
        y[i] = PyFloat_AsDouble(PyList_GetItem(yobj, i));
        yerr[i] = PyFloat_AsDouble(PyList_GetItem(yerrobj, i));
    }
    if (PyErr_Occurred() != NULL)
        return NULL;

    /* Read in the test vector */
    Py_ssize_t m = PyList_Size(x0obj);
    if (PyErr_Occurred() != NULL) {
        PyErr_SetString(PyExc_TypeError, "The target vector must be a list.");
        return NULL;
    }
    VectorXd x0(m);
    for (Py_ssize_t i = 0; i < m; ++i)
        x0[i] = PyFloat_AsDouble(PyList_GetItem(x0obj, i));
    if (PyErr_Occurred() != NULL)
        return NULL;

    double lnlike = 0.0;
    VectorXd mean(1), variance(1);

    int ret = evaluateGP(x, y, yerr, x0, &mean, &variance, &lnlike);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "GP evaluation failed.");
        return NULL;
    }

    /* Build the output */
    PyObject *mean_out = PyList_New(m);
    PyObject *var_out = PyList_New(m);
    if (mean_out == NULL || var_out == NULL) {
        Py_XDECREF(mean_out);
        Py_XDECREF(var_out);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *m = PyFloat_FromDouble(mean[i]);
        PyObject *v = PyFloat_FromDouble(variance[i]);

        if (m == NULL || v == NULL) {
            Py_XDECREF(m);
            Py_XDECREF(v);
            Py_DECREF(mean_out);
            Py_DECREF(var_out);
            return NULL;
        }

        if (PyList_SetItem(mean_out, i, m) != 0 ||
                PyList_SetItem(var_out, i, v) != 0) {
            Py_XDECREF(m);
            Py_XDECREF(v);
            Py_DECREF(mean_out);
            Py_DECREF(var_out);
            return NULL;
        }
    }

    PyObject *tuple_out = Py_BuildValue("OOd", &mean_out, &var_out, &lnlike);
    if (tuple_out == NULL) {
        Py_DECREF(mean_out);
        Py_DECREF(var_out);
        Py_XDECREF(tuple_out);
        return NULL;
    }

    Py_INCREF(tuple_out);
    return tuple_out;
}

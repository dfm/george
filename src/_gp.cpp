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
    PyObject *xobj, *yobj, *yerrobj;
    if (!PyArg_ParseTuple(args, "OOO", &xobj, &yobj, &yerrobj))
        return NULL;

    Py_ssize_t n = PyList_Size(xobj);
    if (n != PyList_Size(yobj) || n != PyList_Size(yerrobj)) {
        PyErr_SetString(PyExc_TypeError, "Dimension mismatch.");
        return NULL;
    }

    VectorXd x(n), y(n), yerr(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        x[int(i)] = PyFloat_AsDouble(PyList_GetItem(xobj, i));
        y[int(i)] = PyFloat_AsDouble(PyList_GetItem(yobj, i));
        yerr[int(i)] = PyFloat_AsDouble(PyList_GetItem(yerrobj, i));
    }

    double lnlike = 0.0;
    VectorXd x0 = VectorXd::LinSpaced(Sequential, 100, -5, 5);
    VectorXd mean(1), variance(1);

    int ret = evaluateGP(x, y, yerr, x0, &mean, &variance, &lnlike);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "GP evaluation failed.");
        return NULL;
    }

    printf("%f\n", lnlike);

    Py_INCREF(Py_None);
    return Py_None;
}

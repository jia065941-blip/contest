#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Exact builtins only. Every unsupported type uses the original SDK copier.
 * Iterators retain Python's callback/mutation behavior. No shared mutable data.
 */
static PyObject *fallback = NULL;

static int atomic(PyObject *v) {
    PyTypeObject *t = Py_TYPE(v);
    return v == Py_None || t == &PyBool_Type || t == &PyLong_Type ||
        t == &PyFloat_Type || t == &PyComplex_Type || t == &PyUnicode_Type || t == &PyBytes_Type;
}

static int keep_alive(PyObject *source, PyObject *memo) {
    PyObject *key = PyLong_FromVoidPtr(memo), *list;
    if (!key) return -1;
    list = PyDict_GetItemWithError(memo, key);
    if (!list) {
        if (PyErr_Occurred()) { Py_DECREF(key); return -1; }
        list = PyList_New(0);
        if (!list) { Py_DECREF(key); return -1; }
        if (PyDict_SetItem(memo, key, list) < 0) { Py_DECREF(list); Py_DECREF(key); return -1; }
        Py_DECREF(list);
        list = PyDict_GetItemWithError(memo, key);
    }
    Py_DECREF(key);
    /* Match copy._keep_alive even for a caller-supplied list subclass. */
    if (PyList_CheckExact(list)) return PyList_Append(list, source);
    PyObject *answer = PyObject_CallMethod(list, "append", "O", source);
    if (!answer) return -1;
    Py_DECREF(answer); return 0;
}

static PyObject *clone_value(PyObject *source, PyObject *memo) {
    PyObject *key = NULL, *cached = NULL, *result = NULL, *iterator = NULL, *item = NULL;
    int is_dict = PyDict_CheckExact(source), is_list = PyList_CheckExact(source);
    if (!atomic(source) && !is_dict && !is_list) return PyObject_CallFunctionObjArgs(fallback, source, memo, NULL);
    key = PyLong_FromVoidPtr(source);
    if (!key) return NULL;
    cached = PyDict_GetItemWithError(memo, key);
    if (cached) { Py_DECREF(key); return Py_NewRef(cached); }
    if (PyErr_Occurred()) { Py_DECREF(key); return NULL; }
    if (atomic(source)) { Py_DECREF(key); return Py_NewRef(source); }
    if (Py_EnterRecursiveCall(" while copying SDK objects")) { Py_DECREF(key); return NULL; }
    result = is_dict ? PyDict_New() : PyList_New(0);
    if (!result || PyDict_SetItem(memo, key, result) < 0) goto fail;
    Py_CLEAR(key);
    if (is_dict) {
        PyObject *items = PyObject_CallMethod(source, "items", NULL);
        if (!items) goto fail;
        iterator = PyObject_GetIter(items);
        Py_DECREF(items);
    } else {
        iterator = PyObject_GetIter(source);
    }
    if (!iterator) goto fail;
    while ((item = PyIter_Next(iterator))) {
        if (is_dict) {
            /* RHS is evaluated before the subscript, as in Python assignment. */
            PyObject *v = clone_value(PyTuple_GET_ITEM(item, 1), memo);
            if (!v) goto fail;
            PyObject *k = clone_value(PyTuple_GET_ITEM(item, 0), memo);
            if (!k) { Py_DECREF(v); goto fail; }
            int ok = PyDict_SetItem(result, k, v);
            Py_DECREF(k); Py_DECREF(v);
            if (ok < 0) goto fail;
        } else {
            PyObject *v = clone_value(item, memo);
            if (!v) goto fail;
            int ok = PyList_Append(result, v);
            Py_DECREF(v);
            if (ok < 0) goto fail;
        }
        Py_CLEAR(item);
    }
    if (PyErr_Occurred() || keep_alive(source, memo) < 0) goto fail;
    Py_DECREF(iterator);
    Py_LeaveRecursiveCall();
    return result;
fail:
    Py_XDECREF(key); Py_XDECREF(item); Py_XDECREF(iterator); Py_XDECREF(result);
    Py_LeaveRecursiveCall();
    return NULL;
}

static PyObject *deepcopy(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *names[] = {"value", "memo", NULL};
    PyObject *source, *memo = Py_None, *answer;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:deepcopy", names, &source, &memo)) return NULL;
    if (!fallback) { PyErr_SetString(PyExc_RuntimeError, "SDK copier not configured"); return NULL; }
    if (memo == Py_None) {
        memo = PyDict_New();
        if (!memo) return NULL;
    } else {
        /* Preserve custom memo mapping protocols through the Python fallback. */
        if (!PyDict_CheckExact(memo)) return PyObject_CallFunctionObjArgs(fallback, source, memo, NULL);
        Py_INCREF(memo);
    }
    answer = clone_value(source, memo);
    Py_DECREF(memo);
    return answer;
}

static PyObject *configure(PyObject *self, PyObject *function) {
    if (!PyCallable_Check(function)) { PyErr_SetString(PyExc_TypeError, "fallback must be callable"); return NULL; }
    Py_XSETREF(fallback, Py_NewRef(function));
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"deepcopy", (PyCFunction)(void(*)(void))deepcopy, METH_VARARGS | METH_KEYWORDS, "Deep copy exact builtin containers; delegate other types."},
    {"configure", configure, METH_O, "Set the original Python copy fallback."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "astra_copy_core", NULL, -1, methods, NULL, NULL, NULL, NULL};
PyMODINIT_FUNC PyInit_astra_copy_core(void) { return PyModule_Create(&module); }

#include <Python.h>
#include "dataset.h"


PyObject * wrap_dset_getbuf (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096];
    static char *kwlist[] = {"dset", "colkey", NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),"s"), kwlist, &dset, &colkey))
    {
        return 0;
    }

    void* mem = 0;
    size_t size = 0;
    Py_BEGIN_ALLOW_THREADS;
    mem = dset_get(dset, colkey);
    size = dset_getsz(dset, colkey);
    Py_END_ALLOW_THREADS;
    return PyMemoryView_FromMemory((char *) mem, (Py_ssize_t) size, PyBUF_WRITE);
}

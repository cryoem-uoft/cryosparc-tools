#include <Python.h>
#include "dataset.h"

#define C2NPY(type) _Generic((type){0},    \
	signed char:        NPY_BYTE,      \
	short:              NPY_SHORT,     \
	int:                NPY_INT,       \
	long:               NPY_LONG,      \
	long long:          NPY_LONGLONG,  \
	unsigned char:      NPY_UBYTE,     \
	unsigned short:     NPY_USHORT,    \
	unsigned int:       NPY_UINT,      \
	unsigned long:      NPY_ULONG,     \
	unsigned long long: NPY_ULONGLONG, \
	float:              NPY_FLOAT,     \
	double:             NPY_DOUBLE,    \
	_Complex float:     NPY_CFLOAT,    \
	_Complex double:    NPY_CDOUBLE    \
	)

#define C2PYFMT(type) _Generic((type){0},  \
	unsigned char:      "b", \
	double:             "d",  \
	float:              "f",  \
	short:              "h",  \
	int:                "i",  \
	long:               "l",  \
	long long:          "L",  \
	unsigned short:     "H",  \
	unsigned int:       "I",  \
	unsigned long:      "k",  \
	unsigned long long: "K"   \
	)

#include <stdarg.h>
#include <stdio.h>
static inline char *
#if defined(__clang__) || defined (__GNUC__)
__attribute__ ((format (printf, 3, 4)))
#endif
fmt_str (int n, char *restrict buffer, const char *restrict fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buffer, n, fmt, ap);
	va_end(ap);
	return buffer;
}

#define NFORMAT(N, fmt, ...) fmt_str(N, (char[N]){0}, (fmt), __VA_ARGS__)
PyObject * wrap_dset_new (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    (void) args;
    (void) kwds;

    uint64_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_new();
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint64_t)), rtn);
} 
PyObject * wrap_dset_del (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    Py_BEGIN_ALLOW_THREADS;
    (void)dset_del(dset);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE; 
} 
PyObject * wrap_dset_copy (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    uint64_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_copy(dset);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint64_t)), rtn);
} 
PyObject * wrap_dset_totalsz (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    uint64_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_totalsz(dset);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint64_t)), rtn);
} 
PyObject * wrap_dset_ncol (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    uint32_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_ncol(dset);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint32_t)), rtn);
} 
PyObject * wrap_dset_nrow (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    uint64_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_nrow(dset);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint64_t)), rtn);
} 
PyObject * wrap_dset_key (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"index",NULL};
    unsigned long dset = {0};
    unsigned long index = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),C2PYFMT(unsigned long)), kwlist, &dset,
		&index )) return 0;


    const char * rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_key(dset, index);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue("s", rtn);
} 
PyObject * wrap_dset_type (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),"s"), kwlist, &dset,
		&colkey )) return 0;


    uint8_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_type(dset, colkey);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint8_t)), rtn);
} 
PyObject * wrap_dset_get (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),"s"), kwlist, &dset,
		&colkey )) return 0;


    void* rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_get(dset, colkey);
    Py_END_ALLOW_THREADS;
    return PyLong_FromVoidPtr(rtn);
} 
PyObject * wrap_dset_getsz (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),"s"), kwlist, &dset,
		&colkey )) return 0;


    uint64_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_getsz(dset, colkey);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint64_t)), rtn);
} 
PyObject * wrap_dset_setstr (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",
		"index",
		"value",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    unsigned long index = {0};
    const char * value = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s%s%s",C2PYFMT(unsigned long),"s",C2PYFMT(unsigned long),"s"), kwlist, &dset,
		&colkey,
		&index,
		&value )) return 0;


    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_setstr(dset, colkey, index, value);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
PyObject * wrap_dset_getstr (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",
		"index",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    unsigned long index = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s%s",C2PYFMT(unsigned long),"s",C2PYFMT(unsigned long)), kwlist, &dset,
		&colkey,
		&index )) return 0;


    const char * rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_getstr(dset, colkey, index);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue("s", rtn);
} 
PyObject * wrap_dset_getshp (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"colkey",NULL};
    unsigned long dset = {0};
    const char * colkey = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),"s"), kwlist, &dset,
		&colkey )) return 0;


    uint32_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_getshp(dset, colkey);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(uint32_t)), rtn);
} 
PyObject * wrap_dset_addrows (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"num",NULL};
    unsigned long dset = {0};
    unsigned int num = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),C2PYFMT(unsigned int)), kwlist, &dset,
		&num )) return 0;


    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_addrows(dset, num);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
PyObject * wrap_dset_addcol_scalar (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"key",
		"type",NULL};
    unsigned long dset = {0};
    const char * key = 0;
    int type = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s%s",C2PYFMT(unsigned long),"s",C2PYFMT(int)), kwlist, &dset,
		&key,
		&type )) return 0;


    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_addcol_scalar(dset, key, type);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
PyObject * wrap_dset_addcol_array (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"key",
		"type",
		"shape0",
		"shape1",
		"shape2",NULL};
    unsigned long dset = {0};
    const char * key = 0;
    int type = {0};
    unsigned char shape0 = {0};
    unsigned char shape1 = {0};
    unsigned char shape2 = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s%s%s%s%s",C2PYFMT(unsigned long),"s",C2PYFMT(int),C2PYFMT(unsigned char),C2PYFMT(unsigned char),C2PYFMT(unsigned char)), kwlist, &dset,
		&key,
		&type,
		&shape0,
		&shape1,
		&shape2 )) return 0;


    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_addcol_array(dset, key, type, shape0, shape1, shape2);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
PyObject * wrap_dset_defrag (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"realloc_smaller",NULL};
    unsigned long dset = {0};
    int realloc_smaller = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s",C2PYFMT(unsigned long),C2PYFMT(int)), kwlist, &dset,
		&realloc_smaller )) return 0;


    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_defrag(dset, realloc_smaller);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
PyObject * wrap_dset_dumptxt (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",NULL};
    unsigned long dset = {0};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s",C2PYFMT(unsigned long)), kwlist, &dset )) return 0;


    Py_BEGIN_ALLOW_THREADS;
    (void)dset_dumptxt(dset);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE; 
} 

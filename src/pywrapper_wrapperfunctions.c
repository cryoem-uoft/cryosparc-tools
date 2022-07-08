uint64_t dset_new ();
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
void dset_del (unsigned long);
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
uint64_t dset_copy (unsigned long);
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
uint64_t dset_totalsz (unsigned long);
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
uint32_t dset_ncol (unsigned long);
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
uint64_t dset_nrow (unsigned long);
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
int8_t dset_type (unsigned long, const char *);
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


    int8_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_type(dset, colkey);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int8_t)), rtn);
} 
void * dset_get (unsigned long, const char *);
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
int dset_setstr (unsigned long, const char *, unsigned long, const char *);
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
const char * dset_getstr (unsigned long, const char *, unsigned long);
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
int dset_addrows (unsigned long, unsigned int);
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
int dset_addcol_scalar (unsigned long, const char *, int);
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
int dset_addcol_array (unsigned long, const char *, int, const uint8_t *);
PyObject * wrap_dset_addcol_array (PyObject *self, PyObject *args, PyObject *kwds)
{
    (void) self;
    char __pyexn_errmsg[4096] = {};
    static char *kwlist[] = {"dset",
		"key",
		"type",
		"shape",NULL};
    unsigned long dset = {0};
    const char * key = 0;
    int type = {0};
    PyArrayObject *shape = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, NFORMAT(1024,"%s%s%s%s",C2PYFMT(unsigned long),"s",C2PYFMT(int),"O!"), kwlist, &dset,
		&key,
		&type,
		&PyArray_Type,&shape )) return 0;
    if(PyArray_TYPE(shape) != C2NPY(uint8_t)){strncpy(__pyexn_errmsg,"Invalid data type for argument shape (expected const uint8_t *)",4095);PyErr_SetString(PyExc_ValueError, __pyexn_errmsg); return 0; }
        if(!PyArray_ISCARRAY(shape)){strncpy(__pyexn_errmsg,"Argument shape is not contiguous",4095);PyErr_SetString(PyExc_ValueError, __pyexn_errmsg); return 0; }

    int rtn = 0;
    Py_BEGIN_ALLOW_THREADS;
    rtn = dset_addcol_array(dset, key, type, PyArray_DATA(shape));
    Py_END_ALLOW_THREADS;
    return Py_BuildValue(NFORMAT(8,"%s",C2PYFMT(int)), rtn);
} 
int dset_defrag (unsigned long, int);
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
void dset_dumptxt (unsigned long);
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

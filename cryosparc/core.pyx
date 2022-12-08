from . cimport dataset
from cython.view cimport array
from cpython.ref cimport PyObject


# Mirror of equivalent C-datatype enumeration
cpdef enum DsetType:
    T_F32 = 1
    T_F64 = 2
    T_C32 = 3
    T_C64 = 4
    T_I8 = 5
    T_I16 = 6
    T_I32 = 7
    T_I64 = 8
    T_U8 = 9
    T_U16 = 10
    T_U32 = 11
    T_U64 = 12
    T_STR = 13
    T_OBJ = 14


cdef class Data:
    cdef dataset.Dset _handle
    cdef dict _strcache

    def __cinit__(self, other = None):
        cdef Data othr
        self._handle = 0
        if isinstance(other, Data):
            # copy constructor
            othr = <Data> other
            self._handle = dataset.dset_copy(othr._handle)
        elif other:
            # Initialize with a numeric handle
            self._handle = <dataset.Dset> other
        else:
            self._handle = dataset.dset_new()

        if self._handle == 0:
            raise MemoryError()

        self._strcache = dict()

    def __dealloc__(self):
        if self._handle:
            dataset.dset_del(self._handle)

    def innerjoin(self, str key, Data other):
        return type(self)(dataset.dset_innerjoin(key.encode(), self._handle, other._handle))

    def totalsz(self):
        return dataset.dset_totalsz(self._handle)

    def ncol(self):
        return dataset.dset_ncol(self._handle)

    def nrow(self):
        return dataset.dset_nrow(self._handle)

    def key(self, int index):
        return dataset.dset_key(self._handle, index).decode()

    def type(self, str field):
        return dataset.dset_type(self._handle, field.encode())

    def has(self, str field):
        return self.type(field) > 0

    def addrows(self, int num):
        return dataset.dset_addrows(self._handle, num)

    def addcol_scalar(self, str field, int dtype):
        return dataset.dset_addcol_scalar(self._handle, field.encode(), dtype)

    def addcol_array(self, str field, int dtype, int shape0, int shape1, int shape2):
        return dataset.dset_addcol_array(self._handle, field.encode(), dtype, shape0, shape1, shape2)

    def getshp(self, str colkey):
        cdef int val = dataset.dset_getshp(self._handle, colkey.encode())
        cdef tuple shape = (val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF)
        return tuple(s for s in shape if s != 0)

    def getbuf(self, str colkey):
        cdef void *mem
        cdef Py_ssize_t size
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        with nogil:
            mem = dataset.dset_get(self._handle, colkey_c)
            size = dataset.dset_getsz(self._handle, colkey_c)
        if size == 0:
            return 0
        else:
            return <unsigned char [:size]> mem

    def getstr(self, str colkey, Py_ssize_t index):
        return dataset.dset_getstr(self._handle, colkey.encode(), index)  # returns bytes

    def tocstrs(self, str colkey):
        # Convert Python strings to C strings in the given object column
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        cdef int prevtype = dataset.dset_type(self._handle, colkey_c)
        cdef Py_ssize_t nrow = dataset.dset_nrow(self._handle)
        cdef PyObject **pycol = <PyObject **> dataset.dset_get(self._handle, colkey_c)
        cdef str pybytes

        if prevtype != T_OBJ or not dataset.dset_changecol(self._handle, colkey_c, T_STR):
            return False

        for i in range(nrow):
            pybytes = <str> (pycol[i])
            pycol[i] = NULL  # so string is not deallocated
            dataset.dset_setstr(self._handle, colkey_c, i, pybytes.encode())

        return True

    def topystrs(self, str colkey):
        # Convert C strings to Python strings in the given column
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        cdef int prevtype = dataset.dset_type(self._handle, colkey_c)
        cdef Py_ssize_t nrow = dataset.dset_nrow(self._handle)
        cdef Py_ssize_t *pycol = <Py_ssize_t *> dataset.dset_get(self._handle, colkey_c)
        cdef void **pystrcol = <void **> pycol
        cdef dict strcache = dict()
        cdef char *cstr
        cdef bytes cbytes
        cdef str pystr

        if prevtype != T_STR:
            return False

        # Save computed strcache to prevent string handles from getting garbage
        # collected (since no other Python reference to them is kept)
        self._strcache[colkey] = strcache

        for i in range(nrow):
            if pycol[i] in strcache:
                pystr = strcache[pycol[i]]
            else:
                cbytes = dataset.dset_getstr(self._handle, colkey_c, i)
                pystr = cbytes.decode()
                strcache[pycol[i]] = pystr

            pystrcol[i] = <void *> pystr

        if not dataset.dset_changecol(self._handle, colkey_c, T_OBJ):
            return False

        return True

    def defrag(self, bint realloc_smaller):
        return dataset.dset_defrag(self._handle, realloc_smaller)

    def dumptxt(self):
        dataset.dset_dumptxt(self._handle)

    def handle(self):
        return self._handle

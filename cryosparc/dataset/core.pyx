from . cimport dataset
from libc.stdint cimport uint8_t, uint16_t, uint64_t
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cpython.mem cimport PyMem_Realloc, PyMem_Free


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

    def __cinit__(self, other = None):
        cdef Data othr
        self._handle = 0
        if isinstance(other, Data):
            # copy constructor
            othr = <Data> other
            self._handle = dataset.dset_copy(othr._handle)
            othr._increfs()
        elif other:
            # Initialize with a numeric handle
            self._handle = <dataset.Dset> other
        else:
            self._handle = dataset.dset_new()

        if self._handle == 0:
            raise MemoryError("Could not allocate dataset")

    def __dealloc__(self):
        if self._handle:
            self._decrefs()
            dataset.dset_del(self._handle)

    def _increfs(self):
        # Increment reference counts for all Python object fields.
        # Call this after making a copy of this dataset.
        nrow = dataset.dset_nrow(self._handle)
        for i in xrange(dataset.dset_ncol(self._handle)):
            with nogil:
                colkey = dataset.dset_key(self._handle, i)
                coltype = dataset.dset_type(self._handle, colkey)
                if coltype != DsetType.T_OBJ:
                    continue
                mem = <PyObject **> (dataset.dset_get(self._handle, colkey))

            for j in xrange(nrow):
                Py_XINCREF(mem[j])

    def _decrefs(self):
        # Decrement reference counts for all Python object fields.
        # Call this after destroying this dataset.
        nrow = dataset.dset_nrow(self._handle)
        for i in xrange(dataset.dset_ncol(self._handle)):
            with nogil:
                colkey = dataset.dset_key(self._handle, i)
                coltype = dataset.dset_type(self._handle, colkey)
                if coltype != DsetType.T_OBJ:
                    continue
                mem = <PyObject **> (dataset.dset_get(self._handle, colkey))

            for j in xrange(nrow):
                Py_XDECREF(mem[j])

    def innerjoin(self, str key, Data other):
        cdef Data data = Data(dataset.dset_innerjoin(key.encode(), self._handle, other._handle))
        data._increfs()
        return data

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
        if not dataset.dset_addrows(self._handle, num):
            raise MemoryError("Could not add rows to dataset")

    def addcol_scalar(self, str field, int dtype):
        if not dataset.dset_addcol_scalar(self._handle, field.encode(), dtype):
            raise MemoryError("Could not add column to dataset")

    def addcol_array(self, str field, int dtype, tuple shape):
        if not 0 < len(shape) <= 3:
            raise ValueError("Shape size must be between 0 and 3")

        cdef uint16_t c_shape[3]
        c_shape[0] = 0; c_shape[1] = 0; c_shape[2] = 0
        for i in xrange(len(shape)):
            c_shape[i] = <uint16_t> shape[i]

        if not dataset.dset_addcol_array(self._handle, field.encode(), dtype, c_shape):
            raise MemoryError("Could not add column to dataset")

    def getshp(self, str colkey):
        cdef list shp = []
        cdef uint64_t val = dataset.dset_getshp(self._handle, colkey.encode())
        cdef uint16_t dim0 = <uint16_t> (val & 0xFFFF)
        cdef uint16_t dim1 = <uint16_t> ((val >> 16) & 0xFFFF)
        cdef uint16_t dim2 = <uint16_t> ((val >> 32) & 0xFFFF)
        if dim0: shp.append(<int> dim0)
        if dim1: shp.append(<int> dim1)
        if dim2: shp.append(<int> dim2)
        return tuple(shp)

    def getbuf(self, str colkey):
        cdef void *mem
        cdef size_t size
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        with nogil:
            mem = dataset.dset_get(self._handle, colkey_c)
            size = dataset.dset_getsz(self._handle, colkey_c)
        return None if size == 0 else <unsigned char [:size]> mem

    def getstr(self, str col, size_t index):
        return dataset.dset_getstr(self._handle, col.encode(), index)  # returns bytes

    def tocstrs(self, str col):
        # Convert Python strings to C strings in the given object column
        cdef bytes colkey = col.encode()
        cdef int prevtype = dataset.dset_type(self._handle, colkey)
        cdef size_t nrow = dataset.dset_nrow(self._handle)
        cdef str pystr
        cdef bytes pybytes

        if prevtype != T_OBJ or not dataset.dset_changecol(self._handle, colkey, T_STR):
            return False

        cdef PyObject **pycol = <PyObject **> dataset.dset_get(self._handle, colkey)
        for i in xrange(nrow):
            pycol = <PyObject **> dataset.dset_get(self._handle, colkey)
            pystr = <str> pycol[i]
            pybytes = pystr.encode()
            Py_XDECREF(pycol[i])  # so string is deallocated
            pycol[i] = NULL  # so that strfree not attempted
            if not dataset.dset_setstr(self._handle, colkey, i, pybytes, len(pybytes)):
                raise MemoryError("Could not convert strings")

        return True

    def topystrs(self, str col):
        # Convert C strings to Python strings in the given column
        cdef bytes colkey = col.encode()
        cdef int prevtype = dataset.dset_type(self._handle, colkey)
        cdef size_t nrow = dataset.dset_nrow(self._handle)
        cdef size_t *pycol = <size_t *> dataset.dset_get(self._handle, colkey)
        cdef PyObject **pystrcol = <PyObject **> pycol
        cdef dict strcache = dict()
        cdef char *cstr
        cdef bytes cbytes
        cdef str pystr

        if prevtype != T_STR:
            return False

        for i in xrange(nrow):
            if pycol[i] in strcache:
                pystr = strcache[pycol[i]]
            else:
                cbytes = dataset.dset_getstr(self._handle, colkey, i)
                pystr = cbytes.decode()
                strcache[pycol[i]] = pystr

            pystrcol[i] = <PyObject *> pystr
            Py_XINCREF(<PyObject *> pystr)

        return bool(dataset.dset_changecol(self._handle, colkey, T_OBJ))

    def dump(self):
        cdef void *mem
        cdef size_t size
        with nogil:
            mem = dataset.dset_dump(self._handle)
            size = dataset.dset_totalsz(self._handle)
        return <unsigned char [:size]> mem

    def dumpstrheap(self):
        cdef void *mem
        cdef size_t size
        with nogil:
            mem = dataset.dset_strheap(self._handle)
            size = dataset.dset_strheapsz(self._handle)
        return <unsigned char [:size]> mem

    def defrag(self, bint realloc_smaller):
        if not dataset.dset_defrag(self._handle, realloc_smaller):
            raise MemoryError("Could not defrag dataset")

    def dumptxt(self, bint dump_data = 0):
        dataset.dset_dumptxt(self._handle, dump_data)

    def handle(self):
        return self._handle

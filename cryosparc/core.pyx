from . cimport dataset
from cython.view cimport array

cdef class Data:
    cdef dataset.Dset _handle

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

    def defrag(self, bint realloc_smaller):
        return dataset.dset_defrag(self._handle, realloc_smaller)

    def dumptxt(self):
        dataset.dset_dumptxt(self._handle)

    def handle(self):
        return self._handle

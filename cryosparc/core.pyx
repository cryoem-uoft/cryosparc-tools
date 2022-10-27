cimport cdataset
from cython.view cimport array

cdef class Data:
    cdef cdataset.Dset _handle

    def __cinit__(self, other = None):
        cdef Data othr
        self._handle = 0
        if isinstance(other, Data):
            # copy constructor
            othr = <Data> other
            self._handle = cdataset.dset_copy(othr._handle)
        elif other:
            # Initialize with a numeric handle
            self._handle = <cdataset.Dset> other
        else:
            self._handle = cdataset.dset_new()

        if self._handle == 0:
            raise MemoryError()

    def __dealloc__(self):
        if self._handle:
            cdataset.dset_del(self._handle)

    def innerjoin(self, str key, Data other):
        return type(self)(cdataset.dset_innerjoin(key.encode(), self._handle, other._handle))

    def totalsz(self):
        return cdataset.dset_totalsz(self._handle)

    def ncol(self):
        return cdataset.dset_ncol(self._handle)

    def nrow(self):
        return cdataset.dset_nrow(self._handle)

    def key(self, int index):
        return cdataset.dset_key(self._handle, index).decode()

    def type(self, str field):
        return cdataset.dset_type(self._handle, field.encode())

    def addrows(self, int num):
        return cdataset.dset_addrows(self._handle, num)

    def addcol_scalar(self, str field, int dtype):
        return cdataset.dset_addcol_scalar(self._handle, field.encode(), dtype)

    def addcol_array(self, str field, int dtype, int shape0, int shape1, int shape2):
        return cdataset.dset_addcol_array(self._handle, field.encode(), dtype, shape0, shape1, shape2)

    def getshp(self, str colkey):
        return cdataset.dset_getshp(self._handle, colkey.encode())

    def getbuf(self, str colkey):
        cdef void *mem
        cdef Py_ssize_t size
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        with nogil:
            mem = cdataset.dset_get(self._handle, colkey_c)
            size = cdataset.dset_getsz(self._handle, colkey_c)
        if size == 0:
            return 0
        else:
            return <unsigned char [:size]> mem

    def defrag(self, bint realloc_smaller):
        return cdataset.dset_defrag(self._handle, realloc_smaller)

    def dumptxt(self):
        cdataset.dset_dumptxt(self._handle)

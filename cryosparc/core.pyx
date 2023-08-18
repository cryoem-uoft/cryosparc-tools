from . cimport dataset
from . cimport lz4
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef int LZ4_ACCELERATION = 1000


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
            raise MemoryError()

    def __dealloc__(self):
        cdef int nrow
        cdef int coltype
        cdef char *colkey
        cdef PyObject **mem
        cdef size_t size
        cdef size_t itemsize
        if not self._handle:
            return

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
            raise MemoryError()

    def reserverows(self, int num):
        if not dataset.dset_reserverows(self._handle, num):
            raise MemoryError()

    def addcol_scalar(self, str field, int dtype):
        return dataset.dset_addcol_scalar(self._handle, field.encode(), dtype)

    def addcol_array(self, str field, int dtype, int shape0, int shape1, int shape2):
        return dataset.dset_addcol_array(self._handle, field.encode(), dtype, shape0, shape1, shape2)

    def extend(self, list others):
        cdef Data other
        cdef dataset.Dset othersarr[256]
        cdef dataset.Dset *othersptr = &othersarr[0]
        cdef int numothers = len(others)
        if numothers > 256:
            othersptr = <dataset.Dset *> PyMem_Malloc(sizeof(dataset.Dset) * numothers)
        try:
            for i in xrange(numothers):
                other = others[i]
                othersptr[i] = other._handle

            if not dataset.dset_extend(self._handle, othersptr, numothers):
                raise MemoryError()

            for i in xrange(numothers):
                other = others[i]
                other._increfs()
        finally:
            if othersptr != &othersarr[0]:
                PyMem_Free(othersptr)

    def getshp(self, str colkey):
        cdef list shp = []
        cdef uint32_t val = dataset.dset_getshp(self._handle, colkey.encode())
        cdef uint8_t dim0 = <uint8_t> (val & 0xFF)
        cdef uint8_t dim1 = <uint8_t> ((val >> 8) & 0xFF)
        cdef uint8_t dim2 = <uint8_t> ((val >> 16) & 0xFF)
        if dim0:
            shp.append(<int> dim0)
        if dim1:
            shp.append(<int> dim1)
        if dim2:
            shp.append(<int> dim2)
        return tuple(shp)

    def getbuf(self, str colkey):
        cdef void *mem
        cdef size_t size
        cdef bytes colkey_b = colkey.encode()
        cdef const char *colkey_c = colkey_b
        with nogil:
            mem = dataset.dset_get(self._handle, colkey_c)
            size = dataset.dset_getsz(self._handle, colkey_c)
        if size == 0:
            return 0
        else:
            return <unsigned char [:size]> mem

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
            dataset.dset_setstr(self._handle, colkey, i, pybytes, len(pybytes))

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

        if not dataset.dset_changecol(self._handle, colkey, T_OBJ):
            return False

        return True

    def stralloc(self, str val):
        cdef uint64_t idx
        cdef bytes pybytes = val.encode()
        if not dataset.dset_stralloc(self._handle, pybytes, len(pybytes), &idx):
            raise MemoryError()
        return <int> idx

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

    def setstrheap(self, bytes heap):
        if not dataset.dset_setstrheap(self._handle, <const char *> heap, len(heap)):
            raise MemoryError()

    def defrag(self, bint realloc_smaller):
        return dataset.dset_defrag(self._handle, realloc_smaller)

    def dumptxt(self, bint dump_data = 0):
        dataset.dset_dumptxt(self._handle, dump_data)

    def handle(self):
        return self._handle


cdef class Stream:
    # Helper class for initializing a dataset from a compressed stream or
    # generating a compressed stream for that dataset.
    #
    # WARNING: Methods here recycle result buffers to minimize allocations. If
    # results are not consumed prior to future calls to this class, the contents
    # will be overwritten.
    cdef Data data
    cdef char *buf  # internal compression buffer
    cdef uint64_t *aux  # internal string index array
    cdef size_t bufsz
    cdef size_t auxsz

    def __cinit__(self, Data data = None):
        self.data = data
        self.buf = NULL
        self.aux = NULL
        self.bufsz = 0
        self.auxsz = 0

    def __dealloc__(self):
        if self.buf != NULL:
            PyMem_Free(self.buf)
            self.buf = NULL
        if self.aux != NULL:
            PyMem_Free(self.aux)
            self.aux = NULL

    def _ensure_buf(self, int min_len):
        cdef size_t sz = min_len
        if self.bufsz < sz:
            self.buf = <char *> PyMem_Realloc(<void *> self.buf, sz)
            self.bufsz = sz
        return <int> self.bufsz

    def _ensure_aux(self, int min_sz):
        cdef size_t sz = min_sz
        if self.auxsz < sz:
            self.aux = <uint64_t *> PyMem_Realloc(<void *> self.aux, sz)
            self.auxsz = sz
        return <int> self.auxsz

    def cast_objs_to_strs(self):
        # change all T_OBJ column types to T_STR (underlying data unaffected)
        cdef dataset.Dset handle = self.data._handle
        cdef int ncol = self.data.ncol()
        cdef int coltype
        cdef const char *colkey

        with nogil:
            for i in xrange(ncol):
                colkey = dataset.dset_key(handle, i)
                coltype = dataset.dset_type(handle, colkey)
                if coltype == T_OBJ:
                    dataset.dset_changecol(handle, colkey, T_STR)

    def stralloc_col(self, str col):
        # Allocate C strings for every Python string in the given object column
        # Returns 0 if the column does not have object type.
        cdef dataset.Dset handle = self.data._handle
        cdef bytes colkey = col.encode()
        cdef int coltype = dataset.dset_type(handle, colkey)
        if coltype != T_OBJ:
            return 0  # invalid column

        # Convert to C strings and compress index array instead
        cdef PyObject **coldata = <PyObject **> dataset.dset_get(handle, colkey)
        cdef uint64_t nrow = dataset.dset_nrow(handle)
        cdef size_t sz = dataset.dset_getsz(handle, colkey)

        self._ensure_aux(sz)

        cdef uint64_t idx
        cdef int allocres
        cdef str pystr
        cdef bytes pybytes

        for i in xrange(nrow):
            pystr = <str> coldata[i]
            pybytes = pystr.encode()
            allocres = dataset.dset_stralloc(handle, pybytes, len(pybytes), &idx)
            if allocres == 0:
                raise MemoryError()
            elif allocres == 2:
                # dataset reallocated, coldata must be retrieved
                coldata = <PyObject **> dataset.dset_get(handle, colkey)
            self.aux[i] = idx

        return <unsigned char [:sz]> (<unsigned char *> self.aux)

    def compress_col(self, str col):
        # Use when multiple compression calls are required to minimize total
        # number of allocations.
        #
        # Overwrites (and potentially re-allocates) internal buffer each time
        # it's called so only the latest return value is valid memory (until the
        # instance is garbage collected).
        #
        # If the column is has type T_OBJ, allocates each python string as
        # a C string in the string heap and compresses the resulting array of
        # indexes into the C string.
        #
        # Returns array memoryview with compressed data
        cdef dataset.Dset handle = self.data._handle
        cdef colkey = col.encode()
        cdef int coltype = dataset.dset_type(handle, colkey)

        if coltype == 0:
            return 0  # invalid column

        cdef uint64_t nrow = dataset.dset_nrow(handle)
        cdef size_t sz = dataset.dset_getsz(handle, colkey)
        cdef uint64_t idx
        cdef int allocres
        cdef str pystr
        cdef bytes pybytes
        cdef PyObject **coldata
        cdef unsigned char [:] data

        if coltype == T_OBJ:
            # Convert to C strings and compress index array instead
            self._ensure_aux(sz)
            coldata = <PyObject **> dataset.dset_get(handle, colkey)

            for i in xrange(nrow):
                pystr = <str> coldata[i]
                pybytes = pystr.encode()
                allocres = dataset.dset_stralloc(handle, pybytes, len(pybytes), &idx)
                if allocres == 0:
                    raise MemoryError()
                elif allocres == 2:
                    # dataset reallocated, coldata must be retrieved
                    coldata = <PyObject **> dataset.dset_get(handle, colkey)
                self.aux[i] = idx

            data = <unsigned char [:sz]> (<unsigned char *> self.aux)
        else:
            data = <unsigned char [:sz]> (<unsigned char *> dataset.dset_get(handle, colkey))

        return self.compress(data)

    def compress_numpy(self, arr):
        cdef size_t sz = arr.size * arr.itemsize
        cdef size_t arr_ptr_val = arr.ctypes.data
        cdef void *arr_ptr = <void *> arr_ptr_val
        return self.compress(<unsigned char [:sz]> arr_ptr)

    def compress(self, unsigned char [:] data):
        cdef int sz = data.size
        cdef int max_compressed_sz = lz4.LZ4_compressBound(sz)

        self._ensure_buf(max_compressed_sz)

         # Call compress. Write final length into compressed_sz
        cdef int compressed_sz = lz4.LZ4_compress_fast(
            <const char *> &data[0],
            self.buf,
            sz,
            max_compressed_sz,
            LZ4_ACCELERATION
        )
        if compressed_sz > 0:
            return <char [:compressed_sz]> self.buf

        raise ValueError(f"Could not compress (error {compressed_sz})")

    def decompress_col(self, str col, bytes data):
        cdef void *mem
        cdef size_t size
        cdef bytes colkey_b = col.encode()
        cdef const char *colkey_c = colkey_b
        with nogil:
            mem = dataset.dset_get(self.data._handle, colkey_c)
            size = dataset.dset_getsz(self.data._handle, colkey_c)
        if mem == NULL:
            raise ValueError(f"Invalid column {col}")
        else:
            return self.decompress(data, <size_t> mem, size)

    def decompress_numpy(self, bytes data, arr):
        cdef size_t dstptr = arr.ctypes.data
        cdef int size = arr.size
        cdef int itemsize = arr.itemsize
        return self.decompress(data, dstptr, size * itemsize)

    def decompress(self, bytes data, size_t dstptr = 0, int dstsz = 0):
        cdef const char *compressed = data
        cdef int compressed_sz = len(data)
        cdef char *uncompressed = <char *> dstptr
        if dstsz <= 0:
            raise ValueError("Decompression buffer size must be > 0")

        if dstptr == 0:
            self._ensure_buf(dstsz)
            uncompressed = self.buf

        cdef int uncompressed_sz = lz4.LZ4_decompress_safe(compressed, uncompressed, compressed_sz, dstsz)
        if uncompressed_sz >= 0:
            return <char [:uncompressed_sz]> uncompressed

        raise ValueError(f"Could not decompress (error {uncompressed_sz})")


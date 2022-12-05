ctypedef Py_ssize_t Dset

cdef extern from "cryosparc-tools/dataset.h":

    Dset dset_new() nogil
    Dset dset_copy(Dset dset) nogil
    Dset dset_innerjoin(const char *key, Dset dset_r, Dset dset_s) nogil
    void dset_del(Dset dset) nogil

    Py_ssize_t dset_totalsz(Dset dset) nogil
    long dset_ncol(Dset dset) nogil
    Py_ssize_t dset_nrow(Dset dset) nogil
    const char *dset_key(Dset dset, Py_ssize_t index) nogil
    int dset_type(Dset dset, const char *colkey) nogil
    void *dset_get(Dset dset, const char *colkey) nogil
    Py_ssize_t dset_getsz(Dset dset, const char *colkey) nogil
    bint dset_setstr(Dset dset, const char *colkey, Py_ssize_t index, const char *value) nogil
    const char *dset_getstr(Dset dset, const char *colkey, Py_ssize_t index) nogil
    long dset_getshp(Dset dset, const char *colkey) nogil

    bint dset_addrows(Dset dset, long num) nogil
    bint dset_addcol_scalar(Dset dset, const char *key, int type) nogil
    bint dset_addcol_array(Dset dset, const char *key, int type, int shape0, int shape1, int shape2) nogil
    bint dset_defrag(Dset dset, bint realloc_smaller) nogil

    void dset_dumptxt(Dset dset) nogil

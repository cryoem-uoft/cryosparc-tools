from libc.stdint cimport uint64_t, uint16_t, uint32_t
ctypedef uint64_t Dset

cdef extern from "cryosparc-tools/dataset.h":

    Dset dset_new() nogil
    Dset dset_copy(Dset dset) nogil
    Dset dset_innerjoin(const char *key, Dset dset_r, Dset dset_s) nogil
    void dset_del(Dset dset) nogil

    uint64_t dset_totalsz(Dset dset) nogil
    uint32_t dset_ncol(Dset dset) nogil
    uint64_t dset_nrow(Dset dset) nogil
    const char *dset_key(Dset dset, uint64_t index) nogil
    int dset_type(Dset dset, const char *colkey) nogil
    void *dset_get(Dset dset, const char *colkey) nogil
    uint64_t dset_getsz(Dset dset, const char *colkey) nogil
    bint dset_setstr(Dset dset, const char *colkey, uint64_t index, const char *value, size_t length) nogil
    const char *dset_getstr(Dset dset, const char *colkey, uint64_t index) nogil
    uint32_t dset_getshp(Dset dset, const char *colkey) nogil

    bint dset_addrows(Dset dset, uint32_t num) nogil
    bint dset_addcol_scalar(Dset dset, const char *key, int type) nogil
    bint dset_addcol_array(Dset dset, const char *key, int type, const uint16_t *shape) nogil
    bint dset_changecol(Dset dset, const char *key, int type) nogil

    bint dset_defrag(Dset dset, bint realloc_smaller) nogil
    void *dset_dump(Dset dset) nogil
    void dset_dumptxt(Dset dset, bint dump_data) nogil

    uint64_t dset_strheapsz(Dset dset) nogil
    char *dset_strheap(Dset dset) nogil
    bint dset_setstrheap(Dset dset, const char *heap, size_t size) nogil
    int dset_stralloc(Dset dset, const char *value, size_t length, uint64_t *index) nogil

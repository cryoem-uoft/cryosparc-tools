cdef extern from "snappy/snappy.h":
    cdef struct snappy_env:
        pass

    int snappy_init_env(snappy_env *env) nogil
    void snappy_free_env(snappy_env *env) nogil
    int snappy_uncompress(const char *compressed, size_t n, char *uncompressed) nogil
    int snappy_compress(
        snappy_env *env,
        const char *input,
        size_t input_length,
        char *compressed,
        size_t *compressed_length
    ) nogil
    bint snappy_uncompressed_length(const char *buf, size_t len, size_t *result) nogil
    size_t snappy_max_compressed_length(size_t source_len) nogil

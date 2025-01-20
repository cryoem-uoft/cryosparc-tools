cdef extern from "lz4/lib/lz4.h":
    int LZ4_compress_default(const char *src, char *dst, int srcSize, int dstCapacity) nogil
    int LZ4_decompress_safe (const char *src, char *dst, int compressedSize, int dstCapacity) nogil
    int LZ4_compressBound(int inputSize) nogil
    int LZ4_compress_fast (const char *src, char *dst, int srcSize, int dstCapacity, int acceleration) nogil

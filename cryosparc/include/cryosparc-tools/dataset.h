#ifndef DATASET_H
#define DATASET_H

#include <complex.h>   // complex number support
#include <stdint.h>    // fixed width integer types
#include <inttypes.h>  // printf specifiers for fixed width integer types

#define repr(x,a,b,val) val 

#ifdef _WIN32
typedef _Fcomplex ds_float_complex_t;
typedef _Dcomplex ds_double_complex_t;
#else
typedef float complex ds_float_complex_t;
typedef double complex ds_double_complex_t;
#endif

#define DSET_TYPELIST(X) \
	X(T_F32,  f,   float,               "f4",   "%g", repr ) \
	X(T_F64,  d,   double,              "f8",   "%g", repr ) \
	X(T_C32,  cf,  ds_float_complex_t,  "c8",   "%s", repr_cfloat ) \
	X(T_C64,  cd,  ds_double_complex_t, "c16",  "%s", repr_cdouble) \
	X(T_I8,   i8,  int8_t,              "i1",   "%" PRIi8,  repr ) \
	X(T_I16,  i16, int16_t,             "i2",   "%" PRIi16, repr ) \
	X(T_I32,  i32, int32_t,             "i4",   "%" PRIi32, repr ) \
	X(T_I64,  i64, int64_t,             "i8",   "%" PRIi64, repr ) \
	X(T_U8,   u8,  uint8_t,             "u1",   "%" PRIu8,  repr ) \
	X(T_U16,  u16, uint16_t,            "u2",   "%" PRIu16, repr ) \
	X(T_U32,  u32, uint32_t,            "u4",   "%" PRIu32, repr ) \
	X(T_U64,  u64, uint64_t,            "u8",   "%" PRIu64, repr ) \
	X(T_STR,  s,   uint64_t,            "O",    "%s", repr_str ) \
	X(T_OBJ,  p,   void*,               "O",    "%p", repr )


enum dset_type {
	T_F32 = 1,
	T_F64 = 2,
	T_C32 = 3,
	T_C64 = 4,
	T_I8  = 5,
	T_I16 = 6,
	T_I32 = 7,
	T_I64 = 8,
	T_U8  = 9,
	T_U16 = 10,
	T_U32 = 11,
	T_U64 = 12,
	T_STR = 13,
	T_OBJ = 14,
};

uint64_t  dset_new (void);
void      dset_del (uint64_t dset);
uint64_t  dset_copy (uint64_t dset);
uint64_t  dset_innerjoin (const char *key, uint64_t dset_r, uint64_t dset_s);

uint64_t    dset_totalsz(uint64_t dset);
uint32_t    dset_ncol   (uint64_t dset);
uint64_t    dset_nrow   (uint64_t dset);
const char* dset_key    (uint64_t dset, uint64_t index);
int         dset_type   (uint64_t dset, const char * colkey);
void *      dset_get    (uint64_t dset, const char * colkey);
uint64_t    dset_getsz  (uint64_t dset, const char * colkey);
int         dset_setstr (uint64_t dset, const char * colkey, uint64_t index, const char * value);
const char* dset_getstr (uint64_t dset, const char * colkey, uint64_t index);
uint32_t    dset_getshp (uint64_t dset, const char * colkey);

int        dset_addrows       (uint64_t dset, uint32_t num);
int        dset_addcol_scalar (uint64_t dset, const char * key, int type);
int        dset_addcol_array  (uint64_t dset, const char * key, int type, int shape0, int shape1, int shape2);
int        dset_changecol     (uint64_t dset, const char * key, int type);

int        dset_defrag (uint64_t dset, int realloc_smaller);
void       dset_dumptxt (uint64_t dset);
void *     dset_dump (uint64_t dset);

#endif
/*
===============================================================================

                 END OF API SECTION, START OF IMPLEMENTATION

   (starts with data and private functions, public API functions at bottom)

===============================================================================
*/
#ifdef DATASET_IMPLEMENTATION

#include <string.h>    // strcmp, memcpy, memmove, memset
#include <stdio.h>     // printf, etc
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>    // functions with variable number of arguments (e.g. error message callback)

#ifdef _WIN32

#define _WIN32_WINNT 0x0600
#include <windows.h>
#define DSALIGNOF __alignof
#define DSNORETURN __declspec(noreturn)
#define DSONCE_INIT INIT_ONCE_STATIC_INIT
#define	DSMUTEX_LOCK_SUCCESS WAIT_OBJECT_0
#define	DSMUTEX_UNLOCK_SUCCESS 1
#define DSMUTEX_LOCK(mutex) WaitForSingleObject(mutex, INFINITE)
#define DSMUTEX_UNLOCK(mutex) ReleaseMutex(mutex)

typedef HANDLE ds_mutex_t;
typedef INIT_ONCE ds_once_t;
typedef	DWORD ds_mutex_lock_t;

#else
#include <stdalign.h>
#include <stdnoreturn.h>
#include <pthread.h>
#define DSALIGNOF alignof
#define DSNORETURN noreturn
#define DSONCE_INIT PTHREAD_ONCE_INIT
#define	DSMUTEX_LOCK_SUCCESS 0
#define	DSMUTEX_UNLOCK_SUCCESS 0
#define DSMUTEX_LOCK(mutex) pthread_mutex_lock(&mutex)
#define DSMUTEX_UNLOCK(mutex) pthread_mutex_unlock(&mutex)

typedef pthread_mutex_t ds_mutex_t;
typedef pthread_once_t ds_once_t;
typedef	int ds_mutex_lock_t;
#endif

/*
	Allow the user to provide a custom allocator if they wish,
	by defining DSREALLOC and DSFREE before including this file
*/
#if defined(DSREALLOC) || defined(DSFREE)
  #if !defined(DSREALLOC) || !defined(DSFREE)
    #error "You must provide both DSREALLOC and DSFREE if you provide either"
  #endif
#else
  #define DSREALLOC realloc
  #define DSFREE    free
#endif




/*
	Allow the user to supply custom error print callbacks if they wish
*/

#ifndef DSPRINTERR
#define DSPRINTERR(str) do{fputs(str, stderr);fflush(stderr);}while(0);
#endif 



/*
	Default error message logging actions
*/
static void 
#if defined(__clang__) || defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif
nonfatal(char *fmt, ...)
{
	char buf[1024];
	char buf2[128];
	char buf3[1024];

	int e = errno;
	if (e != 0) snprintf(buf2,sizeof(buf2)," (errno %d: %s)", e, strerror(e));

	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	snprintf(buf3, sizeof(buf3), "%s%s\n", buf, buf2);
	DSPRINTERR(buf3);
}

static DSNORETURN void
#if defined(__clang__) || defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif
fatal(char *fmt, ...)
{
	char buf[1024];
	char buf2[128];
	char buf3[1024];

	int e = errno;
	if (e != 0) snprintf(buf2,sizeof(buf2)," (errno %d: %s)", e, strerror(e));

	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	snprintf(buf3, sizeof(buf3), "%s%s\n", buf, buf2);
	DSPRINTERR(buf3);
	exit(EXIT_FAILURE);
}

#ifdef NDEBUG
#define xassert(cond) do{(void)(cond);}while(0);
#else
#define xassert(cond) if(!(cond)){ fatal("Assertion failed %s:%i %s", __FILE__, __LINE__, #cond); }
#endif

/*
	Metadata for a single dataset column
*/
typedef struct {
	#define SHORTKEYSZ 52
	union {
		char shortkey[SHORTKEYSZ];
		uint64_t longkey;
	};
	int8_t type; // magnitude matches enum dset_type. negative means use longkey  
	uint8_t shape[3]; // safe to leave as zero for scalars
	uint64_t offset;  // relative to start of respective heap 

} ds_column;


/*
	Metadata for an entire dataset
*/
typedef struct {

	uint8_t    magic[6];
	uint64_t   total_sz; // total allocated memory size
	uint32_t   ccol;  // reserved capacity for columns
	uint32_t   ncol;  // actual number of columns
	uint64_t   crow;  // reserved capacity for rows
	uint64_t   nrow;  // actual number of rows
	uint64_t   arrheap_start; // offset where column data begins
	uint64_t   strheap_start; // offset where string (and other data structure) heap begins
	uint64_t   strheap_sz; // 
	struct {
		// lets track the number of times the more expensive operations occurr
		// in the future we can improve the implementation based on what actually happens a lot
		uint32_t nrealloc;   
		uint32_t nreassign_arroffsets;
		uint32_t nshift_strhandles;
		uint32_t nmore_arrheap;
		uint32_t nmore_strheap;
		uint32_t nmore_colspace;
	} stats;
	ds_column  columns[];

} ds;


/*
	We'll be managing datasets via integer handles instead of via pointers.
	The handle is a composite value containing an index and a "generation" counter.
	The latter is in the upper few bits (15 by default).

	The index is just an index into an array in memory. 
	The generation counter is a way to make sure that if an index gets re-used, 
	we can tell if an old handle that points to this index does indeed refer to this dset.
	This detects the case the user accidentally kept a handle around after deleting a dset.
*/

typedef struct {

	ds         *memory;
	uint16_t   generation;

} ds_slot;


/*
	This is the dataset "module". A single global that stores all the state
	to support datasets.
*/
static struct {
	ds_once_t    init_guard;
	ds_mutex_t   mtx;

	uint64_t          nslots;
	ds_slot *         slots;

} ds_module = {
	.init_guard = DSONCE_INIT,
};

#ifdef _WIN32
BOOL CALLBACK
_module_init(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *lpContext)
{
	// initialize the mutex, and enable some protections against programmer errors
	HANDLE mutex = CreateMutex(
		NULL,  // default security attributes
		FALSE, // initially not owned
		NULL   // unnamed mutex
	);
	xassert(NULL != mutex);
	*lpContext = mutex;
	return TRUE;
}
#else
static void
_module_init(void)
{

	pthread_mutexattr_t a;
	xassert(0 == pthread_mutexattr_init(&a));
	xassert(0 == pthread_mutexattr_settype(&a, PTHREAD_MUTEX_ERRORCHECK));
	xassert(0 == pthread_mutex_init(&ds_module.mtx, &a));
}
#endif


static inline void
module_init(void) {
/*
	make sure that we can call module_init() as often as we want but the underlying init
	happens exactly once.
*/
#ifdef _WIN32
	// Execute the initialization callback function
	xassert(TRUE == InitOnceExecuteOnce(
		&ds_module.init_guard, // One-time initialization structure
		_module_init,          // Pointer to initialization callback function
		NULL,                  // Optional parameter to callback function (not used)
		&ds_module.mtx         // Receives pointer to created mutex
	));
#else
	xassert(0 == pthread_once(&ds_module.init_guard, _module_init));
#endif
}

static inline void
lock (void) {
/*
	This lock only needs to be held when creating or destroying datasets.
	We don't guarantee that datasets can be safely accessed concurrently, that's up to the user.
*/
	ds_mutex_lock_t rc = DSMUTEX_LOCK(ds_module.mtx);
	errno = (int) rc == DSMUTEX_LOCK_SUCCESS;
	xassert(rc == DSMUTEX_LOCK_SUCCESS);
}

static inline void
unlock (void) {
	int rc = DSMUTEX_UNLOCK(ds_module.mtx);
	errno = (int) rc == DSMUTEX_UNLOCK_SUCCESS;
	xassert(rc == DSMUTEX_UNLOCK_SUCCESS);
}


static void 
moreslots (void) {
	static const size_t chunk = 4096;
	void * mem = DSREALLOC(ds_module.slots, sizeof(ds_module.slots[0]) * (ds_module.nslots+chunk));
	if (mem) {
		ds_module.slots   = mem;
		memset(&ds_module.slots[ds_module.nslots], 0, sizeof(ds_module.slots[0])*chunk);
		ds_module.nslots += chunk;
	}
}

#define SHIFT_GEN (64-16)
#define MASK_IDX  (0xffffffffffffffff >> 16)
#define MAX_GEN   UINT16_MAX

static inline uint64_t
roundup(uint64_t value, uint64_t to)
{
	return value+to-(value%to);
}

static uint64_t 
dset_new_(size_t newsize, ds **allocation) 
{
	module_init();
	lock();

	ds_slot *s;
	uint64_t gen;
	void *mem;

	// see if we can find an existing empty slot
	uint64_t i = 0;
	for (i = 0; i < ds_module.nslots; i++) {

		if (! ds_module.slots[i].memory) break;
	}

	if (i == ds_module.nslots)
		moreslots();

	if (i == ds_module.nslots) 
		goto out_of_memory;


	s = &ds_module.slots[i];

	mem = DSREALLOC(0, newsize);
	if (!mem) goto out_of_memory;

	*allocation = (ds *) mem;
	s->memory   = (ds *) mem;
	unlock();

	memset(s->memory, 0, newsize);
	if (s->generation >= MAX_GEN) {
		// Generation limit reached, trigger overflow so that handle != 0
		s->generation = 0;
	}
	gen = ++s->generation;

	return i | (gen << SHIFT_GEN);
	 
	out_of_memory:
	unlock();
	nonfatal("out of memory");
	return UINT64_MAX;
}



static ds* 
handle_lookup (uint64_t h, const char * msg_fragment, uint16_t * gen, uint64_t * idx) 
{
	uint16_t gen_ = 0;
	uint64_t idx_ = 0;

	// allow passing null pointers for gen and idx
	idx = idx ? idx : &idx_;
	gen = gen ? gen : &gen_;

	*idx = MASK_IDX & h;
	*gen = h >> SHIFT_GEN;

	if (ds_module.nslots <= *idx) {
		nonfatal("%s: invalid handle %" PRIu64 ", no such slot", msg_fragment, h);
		return 0;
	}

	if (!ds_module.slots[*idx].memory) { 
		nonfatal("%s: invalid handle %" PRIu64 ", no heap at index %" PRIu64, msg_fragment, h, *idx);
		return 0;
	}

	if (ds_module.slots[*idx].generation != *gen) {
		nonfatal("%s: invalid handle %" PRIu64 ", wrong generation counter"
				" (given %" PRIu16 ", expected %" PRIu16")", 
				msg_fragment, h, *gen, ds_module.slots[*idx].generation);
		return 0;
	}


	return ds_module.slots[*idx].memory;
}






#define XCONCAT(a,b) a ## b
#define CONCAT(a,b) XCONCAT(a,b)


#define EMIT_TYPEENUM_ONLY(typeenum,a,b,c,d,e) typeenum,
static const 
int valid_types[] = { DSET_TYPELIST(EMIT_TYPEENUM_ONLY) };

static const 
size_t Ntypes = sizeof(valid_types)/sizeof(valid_types[0]);

#define EMIT_SZ_ARRAY_ENTRY(typeenum,a,ctype,b,c,d) [typeenum]=sizeof(ctype),
static const 
size_t type_size[] = { DSET_TYPELIST(EMIT_SZ_ARRAY_ENTRY) };

#define EMIT_ALIGNCHECK(a,name,ctype,b,c,d) static_assert(DSALIGNOF(ctype) <= 16, "platform incompatible");
DSET_TYPELIST(EMIT_ALIGNCHECK) ;


static inline int8_t abs_i8 (int8_t x) {return x < 0 ? -x : x;}
#define EMIT_TYPECHECK_FUNCTION(typeenum, fnsuffix, a,b,c,d) \
	static int CONCAT(tcheck_, fnsuffix) (int8_t type) {     \
		return abs_i8(type) == typeenum;    \
	}
DSET_TYPELIST(EMIT_TYPECHECK_FUNCTION)

static int
tcheck(int8_t type)
{
	const int t  = abs_i8(type);
	
	for (int i = 0; i < Ntypes; i++) 
		if (t == valid_types[i]) 
			return 1;

	return 0;
}


static inline size_t
stride (const ds_column *c)
{
	// returns stride in units of the datatype size (not in bytes)!
	size_t s = 1;
	s *= c->shape[0] ? c->shape[0] : 1;
	s *= c->shape[1] ? c->shape[1] : 1;
	s *= c->shape[2] ? c->shape[2] : 1;
	return s;
}

static const char *
getkey(const ds *d, const ds_column *c) 
{
	char * ptr = (char *)d;
	const char * key = (c->type < 0) ? ptr + d->strheap_start + c->longkey : c->shortkey;
	return key;
}

static ds_column * 
column_lookup(ds * d, const char * colkey)
{
	if(!d) return 0;

	char * ptr = (char *)d;
	ds_column *c = d->columns;
	long i = 0;
	for (;  i < d->ncol;  i++,c++) {
		const char * key = getkey(d, c);
		if (!strcmp(key, colkey)) return c;
	}

	// nonfatal("key error: %s", colkey);
	return 0;
}


static inline uint64_t 
arrheap_capacity (const ds *d) {
	return d->strheap_start - d->arrheap_start;
}

static inline uint64_t
strheap_capacity (const ds *d) {
	return d->total_sz - d->strheap_start;
}

static inline uint64_t
compute_col_reserved_space (uint32_t crow, const ds_column *c) {

	const uint64_t col_stride = type_size[abs_i8(c->type)] * stride(c);
	return roundup(crow*col_stride, 16);
}

static ds*
more_memory (uint64_t idx, uint64_t nbytes_more) {

	const uint64_t more = roundup(nbytes_more, 1<<15); // 32 kB at a time (too little? too much?)

	ds *d = ds_module.slots[idx].memory;
	d->stats.nrealloc++;

	ds * newptr = DSREALLOC(d, d->total_sz + more);
	if (!newptr) {
		nonfatal("dataset.more_memory: out of memory");
		return 0;
	}

	ds_module.slots[idx].memory = d = newptr;

	char * ptr = (char *) newptr;
	memset(ptr + d->total_sz, 0, more);

	d->total_sz += more;

	return d;
}

static ds* 
more_strheap (uint64_t idx, uint64_t nbytes_more) {

	ds *d = ds_module.slots[idx].memory;
	d->stats.nmore_strheap++;

	uint64_t arrheap_reqdsize = 0;
	if (d->ncol > 0) {
		arrheap_reqdsize   =  d->columns[d->ncol-1].offset;
		arrheap_reqdsize  +=  compute_col_reserved_space(d->crow, &d->columns[d->ncol-1]);
	}

	uint64_t arrheap_actualsize = d->strheap_start - d->arrheap_start;

	// if we can find the space just by shrinking the array heap, do that.
	if (arrheap_actualsize - arrheap_reqdsize >= nbytes_more) {

		char * ptr = (char *) d;
		char * move_src = ptr + d->strheap_start;
		char * move_dst = move_dst - nbytes_more;

		memmove (move_dst, move_src, d->strheap_sz);
		memset  (move_dst + d->strheap_sz, 0, d->strheap_sz);

		d->strheap_start -= nbytes_more;
		return d;
	}

	// otherwise we need more memory
	return more_memory(idx, nbytes_more);
}

static ds*
more_arrheap (uint64_t idx, uint64_t nbytes_more) {

	ds *d = ds_module.slots[idx].memory;
	d->stats.nmore_arrheap++;

	do {
		// if we can find the space just by shrinking the string heap, do that.
		if (d->total_sz - d->strheap_start - d->strheap_sz   >   nbytes_more) {

			char * ptr = (char *) d;
			char * move_src = ptr + d->strheap_start;
			char * move_dst = move_src + nbytes_more;

			memmove (move_dst, move_src, d->strheap_sz);
			memset  (move_src, 0,        d->strheap_sz);

			d->strheap_start += nbytes_more;
			return d;
		}

		// otherwise we need more memory
		d = more_memory(idx, nbytes_more);
	} while(d);
	return 0;
}

static ds*
more_columndescr_space (uint64_t idx, uint64_t ncolumns_more) {

	uint64_t nbytes_more = ncolumns_more * sizeof(ds_column);

	ds *d = ds_module.slots[idx].memory;
	d->stats.nmore_colspace++;

	// for simplicity, let's not steal from the array heap, just from the string heap
	do {
		// if we can find the space just by shrinking the string heap, do that.
		if (d->total_sz - d->strheap_start - d->strheap_sz   >   nbytes_more) {

			char * move_src =(char *)  &d->columns[d->ccol];
			char * move_dst = move_src + nbytes_more;
			uint64_t arrheap_size = d->strheap_start - d->arrheap_start;

			memmove (move_dst, move_src, arrheap_size + d->strheap_sz);
			memset  (move_src, 0,        nbytes_more);

			d->strheap_start += nbytes_more;
			d->arrheap_start += nbytes_more;
			d->ccol += ncolumns_more;

			return d;
		}

		// otherwise we need more memory
		d = more_memory(idx, nbytes_more);
	} while(d);
	return 0;

}

static uint64_t
actual_arrheap_sz (ds *d) {
	if (d->ncol > 0) {
		ds_column last = d->columns[d->ncol-1];
		return compute_col_reserved_space(d->crow, &last) + last.offset;
	}
	return 0;
}

/**
 * Definitions for hash table where every key and value is 64 bits
 * Based on https://nullprogram.com/blog/2022/08/08/
 * Future work: Adapt for generic key/value types for the string buffer.
 */

// Invalid or unset hashtable entry (64 bit all ones)
#define DSHT64_INVALID 0xffffffffffffffffU

typedef uint64_t ds_ht64_row[2];

/**
 * Numeric hashtable where each key and value is a 64 bit integer
 */
typedef struct ds_ht64 {
	ds_ht64_row *ht; // array of key/value pairs
	int32_t len;
	int32_t exp; // exponent that denotes total hashtable capacity
} ds_ht64;

// https://nullprogram.com/blog/2018/07/31/
static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    return x;
}

// Allocate a hashtable with the given size
static void ht64_new(ds_ht64 *t, uint32_t sz) {
	// Allocate the given size for the hash table t
	uint32_t exp = 0;
	do { exp++; } while ((1 << exp) < sz);
	exp += 1; // at least double the required size for fast lookups

	size_t totalsz = sizeof(ds_ht64_row) * (1 << exp);
	void *mem = DSREALLOC(0, totalsz);
	// Initialize memory to -1 (all ones)
	memset(mem, -1, totalsz);
	t->ht = mem;
	xassert(t->ht[0][0] == DSHT64_INVALID) // verify that memset worked
	t->len =  0;
	t->exp = exp;
}

// Free an allocated hash table
static void ht64_del(ds_ht64 *t) {
	if (t->ht) {
	DSFREE(t->ht);
	}
	t->ht = 0;
	t->len = 0;
	t->exp = 0;
}

// Compute the next candidate index. Initialize idx to the hash.
static inline int32_t ht64_lookup(uint64_t hash, int exp, int32_t idx) {
	uint32_t mask = ((uint32_t)1 << exp) - 1;
	uint32_t step = (hash >> (64 - exp)) | 1;
	return (idx + step) & mask;
}

static inline int ht64_has(ds_ht64 *t, const uint64_t key) {
	uint64_t h = hash64(key);
	for (int32_t i = h;;) {
		i = ht64_lookup(h, t->exp, i);
		if (t->ht[i][0] == DSHT64_INVALID) return 0; // empty, does not exist
		else if (t->ht[i][0] == key) return 1; // found
		// otherwise keep looking
	}
	return 0;
}

// Find the value of key in the hash table. Put the result in val. Returns 1
// (true) if located, 0 otherwise.
static int ht64_find(ds_ht64 *t, const uint64_t key, uint64_t *val) {
	uint64_t h = hash64(key);
	for (int32_t i = h;;) {
		i = ht64_lookup(h, t->exp, i);
		if (t->ht[i][0] == DSHT64_INVALID) {
			// empty
			return 0;
		} else if (t->ht[i][0] == key) {
			// Found, populate result
			*val = t->ht[i][1];
			return 1;
		}
		// Otherwise keep looking
	}
	return 0;
}

// Insert a value into the hash table
// Will overwrite val if the key already exists
static int ht64_insert(ds_ht64 *t, const uint64_t key, const uint64_t val) {
	if ((uint32_t) t->len == (uint32_t) (1 << t->exp)) {
		// hash table is full
		return 0;
	}

	uint64_t h = hash64(key);
	for (int32_t i = h;;) {
		i = ht64_lookup(h, t->exp, i);
		if (t->ht[i][0] == DSHT64_INVALID || t->ht[i][0] == key) {
			// empty or existing key, insert here
			t->len++;
			t->ht[i][0] = key;
			t->ht[i][1] = val;
			return 1;
		}
		// Otherwise keep looking for a spot to insert
	}
	return 0;
}


static uint64_t
stralloc(ds **d, uint64_t idx, const char * str)
{
	const size_t sz = 1 + strlen(str);
	char * base = (char*)*d;

	// do we already have this string?
	{
		char * strheap = base + (*d)->strheap_start;
		char * strheap_end = strheap + (*d)->strheap_sz;

		char * p = strheap;
		while (p < strheap_end) {
			if(!strcmp(str,p)) return p-strheap;
			p += strlen(p) + 1;
		}
	} // guess not...

	// do we need more space?
	if ((*d)->total_sz - (*d)->strheap_start < (*d)->strheap_sz + sz) {
		*d = more_strheap(idx, sz);
		if (!*d) return 0;
	}

	uint64_t newstr   = (*d)->strheap_sz;
	(*d)->strheap_sz += sz;

	memcpy(base + (*d)->strheap_start + newstr, str, sz);
	return newstr; 
}

static void 
shift_all_string_handles(ds *d, int64_t shift, uint64_t shift_greater_than)
{
	char * ptr = (char *) d;
	for (uint32_t i = 0; i < d->ncol; i++) {
		if (d->columns[i].type < 0   &&   d->columns[i].longkey > shift_greater_than) {
			d->columns[i].longkey += shift;
		}

		if (tcheck_s(d->columns[i].type)) {
			uint64_t *handles = ptr + d->arrheap_start + d->columns[i].offset;
			for (uint64_t j = 0 ; j < d->nrow; j++) {
			
				if (handles[j] > shift_greater_than)
					handles[j] += shift;
			}
		}
	}
	d->stats.nshift_strhandles++;
}

static void
strfree (uint64_t oldstr, ds *d)
{
	// TODO: This should check that no one else is using this string before freeing it
	if (!oldstr) return;
	char * strheap = ((char *)d) + d->strheap_start;
	char * s = strheap + oldstr;
	int64_t sz = 1 + strlen(s);
	memmove(s, s+sz, (strheap+d->strheap_sz) - (s+sz));
	shift_all_string_handles(d, -sz, oldstr);
	d->strheap_sz -= sz;
}

static inline char *
getstr(ds *d, ds_column *c, uint64_t index) {
	char * ptr = (char *) d;
	uint64_t *handles = ptr + d->arrheap_start + c->offset;
	return ptr + d->strheap_start + handles[index];
}

// Set string helper that returns dataset pointer with string assigned (may be
// the same dataset pointer or different if required reallocation)
static ds *
setstr (ds *d, ds_column *c, uint64_t index, const char *value) {
	uint64_t idx;
	char *ptr = (char *) d;
	uint64_t *handles = ptr + d->arrheap_start + c->offset;
	strfree(handles[index], d);
	uint64_t newstr = stralloc(&d, idx, value);
	if (!d) return 0; // Could not allocate string

	handles[index] = newstr;
	return d;
}

static void
reassign_arrayoffsets (ds *d,  uint32_t new_crow)
{
	uint64_t cur_arrheap_used_sz = actual_arrheap_sz(d);

	char * arrheap = ((char *)d) + d->arrheap_start;
	char * arrheap_end = arrheap + cur_arrheap_used_sz;

	for(uint32_t i = 1; i < d->ncol; i++) {

		ds_column * lastcol = d->columns+i-1;

		const size_t lastcol_oldsz = compute_col_reserved_space(d->crow, lastcol);
		const size_t lastcol_newsz = compute_col_reserved_space(new_crow, lastcol);

		const ptrdiff_t shift   = lastcol_newsz - lastcol_oldsz;
		char * mov_src = arrheap + lastcol_oldsz + lastcol->offset;
		char * mov_dst = mov_src + shift;

		// since we're forwards-iterating, if we forward shift, we need to move the entire array heap
		const ptrdiff_t nbytes  = (shift > 0)  ?  (arrheap_end - mov_src)  :  lastcol_oldsz;

		memmove (mov_dst, mov_src, nbytes);
		arrheap_end += shift;

		if (shift > 0) memset (mov_src, 0, mov_dst-mov_src);

		d->columns[i].offset = mov_dst-arrheap;
	}

	d->stats.nreassign_arroffsets++;
}

static inline void
copyval(
	ds *dst_ds, ds_column *dst_col, uint64_t dst_idx,
	ds *src_ds, ds_column *src_col, uint64_t src_idx,
	size_t itemsize // assume same stride on each column
) {
	char *dst_ptr = (char *) dst_ds + dst_ds->arrheap_start + dst_col->offset + (dst_idx * itemsize);
	char *src_ptr = (char *) src_ds + src_ds->arrheap_start + src_col->offset + (src_idx * itemsize);
	memcpy(dst_ptr, src_ptr, itemsize);
}

static inline ds *
copystr(
	ds *dst_ds, ds_column *dst_col, uint64_t dst_idx,
	ds *src_ds, ds_column *src_col, uint64_t src_idx
) {
	char *str = getstr(src_ds, src_col, src_idx);
	return setstr(dst_ds, dst_col, dst_idx, str);
}

/*
===============================================================================
                           ACTUAL API FUNCTIONS
===============================================================================
*/



uint64_t dset_new(void) {
	const size_t DS_INITIAL_SZ = 1<<15; // 32 kB as a good default?
	ds * d = 0;

	uint64_t handle = dset_new_(DS_INITIAL_SZ, &d);
	if(handle == UINT64_MAX) 
		return handle;

	*d = (ds) {
		.total_sz = DS_INITIAL_SZ,
		.arrheap_start = sizeof(*d),
		.strheap_start = sizeof(*d),
		.strheap_sz    = 1, // the null string is the string with index zero.
	};

	// 0x94 CSDAT
	d->magic[0] = 0x94;
	d->magic[1] = 0x43;
	d->magic[2] = 0x53;
	d->magic[3] = 0x44;
	d->magic[4] = 0x41;
	d->magic[5] = 0x54;

	return handle;
}

uint64_t dset_copy(uint64_t dset)
{
	uint64_t idx;
	uint16_t generation;

	if(! handle_lookup(dset, "dset_del", &generation, &idx))
		return UINT64_MAX;

	ds *oldds = ds_module.slots[idx].memory;

	ds* newds = 0;
	uint64_t newhandle = dset_new_(oldds->total_sz, &newds);

	if (newhandle != UINT64_MAX) 
		memcpy(newds,oldds,oldds->total_sz); 

	return newhandle;
}

// Compute the inner join of two Datasets R and S by matching values in the
// column with the given key. Currently only 64-bit columns (e.g., T_U64) with
// zero shape may be specified as keys.
//
// Not recommended for key columns with duplicate values (does not deduplicate
// and only matches one row in joined dataset).
//
// Implements Classic Hash Join algorithm
// https://en.wikipedia.org/wiki/Hash_join#Classic_hash_join
typedef struct ds_innerjoin_coldata {
	ds_column *col;
	int itemsize;
	int is_str;
} ds_innerjoin_coldata;

uint64_t dset_innerjoin(const char *key, uint64_t dset_r, uint64_t dset_s)
{
	uint64_t dset = 0;
	uint64_t idx_r, idx_s;
	uint16_t generation_r, generation_s;
	ds *ds_r, *ds_s;
	ds_column *keycol_r, *keycol_s;
	ds_ht64 idx_lookup; idx_lookup.ht = 0;
	uint64_t *keydata_r, *keydata_s;
	uint32_t nrow = 0;

	// Look up the two datasets and columns to join
	if (!(ds_r = handle_lookup(dset_r, "dset_innerjoin", &generation_r, &idx_r))) return UINT64_MAX;
	if (!(ds_s = handle_lookup(dset_s, "dset_innerjoin", &generation_s, &idx_s))) return UINT64_MAX;
	keycol_r = column_lookup(ds_r, key);
	keycol_s = column_lookup(ds_s, key);

	if (!keycol_r || !keycol_s) {
		nonfatal("dset_innerjoin: input dataset does not contain %s column", key);
		return UINT64_MAX;
	}
	if (keycol_r->type != keycol_s->type) {
		nonfatal("dset_innerjoin: input %s column types do match (%d, %d)", key, keycol_r->type, keycol_s->type);
		return UINT64_MAX;
	}
	if (keycol_r->shape[0] != 0 || keycol_s->shape[0] != 0) {
		nonfatal("dset_innerjoin: cannot innerjoin column %s with non-zero shape", key);
		return UINT64_MAX;
	}

	if (keycol_r->type != T_U64 && keycol_r->type != T_I64 && keycol_r->type != T_F64 && keycol_r->type != T_C32) {
		// TODO: Allow innerjoining any type (or least numeric types)
		nonfatal("dset_innerjoin: cannot innerjoin column %s with non-64bit type %d", key, keycol_r->type);
		return UINT64_MAX;
	}

	// Allocate new dataset with unioned fields
	dset = dset_new();

	// Populate fields from the first dataset R. Declare a stack-allocated
	// dynamic array of structs which memoize the required column data
	ds_column *col;
	char *colkey;

	// Cache source column details (try to use stack version if possible)
	ds_innerjoin_coldata src_coldata_stack[1024];
	ds_innerjoin_coldata *src_coldata = src_coldata_stack;
	if (ds_r->ncol + ds_s->ncol > 1024) {
		src_coldata = DSREALLOC(0, sizeof(ds_innerjoin_coldata) * (ds_r->ncol + ds_s->ncol));
	}

	uint32_t nrcol = 0, nscol = 0; // number of columns used from R and S
	for (uint32_t c = 0; c < ds_r->ncol; c++) {
		colkey = dset_key(dset_r, c);
		if (strcmp(key, colkey) == 0 || !column_lookup(ds_s, colkey)) {
			// key is either target join key or not in other dataset, add now
			// with correct type details
			col = column_lookup(ds_r, colkey);
			if (!dset_addcol_array(
				dset, colkey, col->type,
				col->shape[0], col->shape[1], col->shape[2]
			)) {
				nonfatal("dset_innerjoin: cannot add column %s to result dataset", colkey);
				goto fail;
			}
			src_coldata[nrcol].col = col;
			src_coldata[nrcol].itemsize = type_size[abs_i8(col->type)] * stride(col);
			src_coldata[nrcol].is_str = col->type == T_STR;
			nrcol++;
		} // otherwise defer to dataset S for this column
	}

	// Populate fields from the second dataset S
	for (uint32_t c = 0; c < ds_s->ncol; c++) {
		colkey = dset_key(dset_s, c);
		if (strcmp(key, colkey) == 0) {
			continue; // already added in previous loop
		}
		col = column_lookup(ds_s, colkey);
		if (!dset_addcol_array(
			dset, colkey, col->type,
			col->shape[0], col->shape[1], col->shape[2]
		)) {
			nonfatal("dset_innerjoin: cannot add column %s to result dataset", colkey);
			goto fail;
		}
		src_coldata[nrcol + nscol].col = col;
		src_coldata[nrcol + nscol].itemsize = type_size[abs_i8(col->type)] * stride(col);
		src_coldata[nrcol + nscol].is_str = col->type == T_STR;
		nscol++;
	}

	// Get the data for each column to join
	keydata_r = dset_get(dset_r, key);
	keydata_s = dset_get(dset_s, key);

	// Create a hash table of values which store the index of each column value
	// in dataset S
	ht64_new(&idx_lookup, ds_s->nrow);
	for (uint64_t j = 0; j < ds_s->nrow; j++) {
		if (!ht64_insert(&idx_lookup, keydata_s[j], j)) {
			nonfatal("dset_innerjoin: hash table full?? cannot proceed");
			goto fail;
		}
	}

	// Determine resulting number of rows in joined dataset. Note that entries
	// in dataset R that share the same value will not be de-duplicated and only
	// the highest index matching row in dataset S will be used.
	for (uint64_t i = 0; i < ds_r->nrow; i++) {
		if (ht64_has(&idx_lookup, keydata_r[i])) nrow++;
	}
	dset_addrows(dset, nrow);

	// Populate columns of new dataset
	uint64_t idx;
	uint16_t generation;
	ds *d;
	uint32_t c;

	if (!(d = handle_lookup(dset, "dset_innerjoin", &generation, &idx))) return UINT64_MAX;

	// Indeces i corresponds to R indices, j to S indices, k to the result dataset
	for (uint64_t i = 0, k = 0, j = 0; i < ds_r->nrow; i++) {
		if (!ht64_find(&idx_lookup, keydata_r[i], &j)) {
			continue; // not in dataset, don't populate it
		}

		// Copy row values from Dataset R
		for (c = 0; c < nrcol; c++) {
			if (src_coldata[c].is_str) {
				d = copystr(d, &d->columns[c], k, ds_r, src_coldata[c].col, i);
			} else {
				copyval(d, &d->columns[c], k, ds_r, src_coldata[c].col, i, (size_t) src_coldata[c].itemsize);
			}
		}

		// Copy row values from Dataset S
		for (c = nrcol; c < nrcol + nscol; c++) {
			if (src_coldata[c].is_str) {
				d = copystr(d, &d->columns[c], k, ds_s, src_coldata[c].col, j);
			} else {
				copyval(d, &d->columns[c], k, ds_s, src_coldata[c].col, j, (size_t) src_coldata[c].itemsize);
			}
		}

		k++; // increment row
	}

	// Success! Skip over the fail case, cleanup and return the handle
	goto done;

	fail:
	// Delete and invalidate dataset
	if (dset) dset_del(dset);
	dset = UINT64_MAX;

	done:
	// Clean up hash table
	// Free up memoized column data, if necessary
	ht64_del(&idx_lookup);
	if (src_coldata != src_coldata_stack) DSFREE(src_coldata);

	return dset;
}

void dset_del(uint64_t dset)
{
	module_init();
	lock();

	uint64_t idx;
	uint16_t generation;
	if (handle_lookup(dset, "dset_del", &generation, &idx)) {

		DSFREE(ds_module.slots[idx].memory);
		ds_module.slots[idx].memory = 0;
	}
	unlock();
}

uint64_t dset_totalsz(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_ncol", 0, 0);
	if(d) return d->total_sz;
	else  return 0;
}

uint32_t dset_ncol(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_ncol", 0, 0);
	if(d) return d->ncol;
	else  return 0;
}

uint64_t dset_nrow(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_nrow", 0, 0);
	if(d) return d->nrow;
	else  return 0;
}

const char *dset_key(uint64_t dset, uint64_t index)
{
	const ds *d  = handle_lookup(dset, "dset_colkey", 0, 0);
	if (!d) return "";
	if (index >= d->ncol) {
		nonfatal("dset_key: column index %d out of range (%d ncol)", index, d->ncol);
		return "";
	}
	const ds_column *c  = &d->columns[index];
	return getkey(d, c);
}


int dset_type (uint64_t dset, const char * colkey)
{
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);

	if(!(d && c)) return 0;
	return abs_i8(c->type);
}

void *dset_get (uint64_t dset, const char * colkey)
{
	// Caution: T_STR columns cannot be used directly, actual strings must be
	// retrieved through dset_getstr
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);
	char * ptr = (char *) d;

	if(!(d && c)) return 0;

	return ptr + d->arrheap_start + c->offset;
}

uint64_t dset_getsz(uint64_t dset, const char * colkey)
{
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);

	if(!(d && c)) return 0;

	return d->nrow * abs_i8(type_size[c->type]) * stride(c);
}

uint32_t dset_getshp (uint64_t dset, const char * colkey)
{
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);

	if(!(d && c)) return 0;

	// Each byte in the result is a member of the shape tuple (ordered by
	// significance)
	return c->shape[0] | c->shape[1] << 8 | c->shape[2] << 16;
}

int dset_addcol_scalar (uint64_t dset, const char * key, int type) {
	return dset_addcol_array(dset, key, type, 0, 0, 0);
}


int dset_addcol_array (uint64_t dset, const char * key, int type, int shape0, int shape1, int shape2) {

	if(!tcheck(type)) {
		nonfatal("invalid column data type: %i", type);
		return 0;
	}

	uint64_t idx; 
	ds *d = handle_lookup(dset, "add column", 0, &idx);
	if(!d) return 0;

	const size_t ksz = 1 + strlen(key);
	const int8_t t   = abs_i8(type);

	// hypothetical new column descriptor.
	ds_column col;
	col.type =  ksz > SHORTKEYSZ ? -t : t;
	col.shape[0] = (uint8_t) shape0;
	col.shape[1] = (uint8_t) shape1;
	col.shape[2] = (uint8_t) shape2;

	if (d->ncol == d->ccol) {

		// need more space for column descriptors
		d = more_columndescr_space(idx, 30);
		if (!d) return 0;
	}

	// compute the offset for the new column
	col.offset = actual_arrheap_sz(d);
	
	// compute the required space on the array heap for the  new column
	const uint64_t reqd_arrheap_space = compute_col_reserved_space(d->crow, &col);

	const uint64_t arrheap_sz = d->strheap_start - d->arrheap_start;

	if (col.offset + reqd_arrheap_space > arrheap_sz) {

		// need more space on the array heap
		const size_t howmuch_need = col.offset + reqd_arrheap_space - arrheap_sz;
		d = more_arrheap(idx, howmuch_need);
		if (!d) return 0;
	}

	if (col.type < 0) {
		// key is long, so allocate a string for it	

		uint64_t newstr = stralloc(&d, idx, key);
		if (!d) return 0;
		col.longkey    = newstr;

	} else {
		snprintf(col.shortkey, sizeof(col.shortkey), "%s", key);
	}

	// commit the new column
	d->columns[d->ncol++] = col;
	return 1;
}

// Change the type of the given column. Type must be compatible in size
// NOTE: This is unsafe! Does not cast existing values to expected values
int dset_changecol (uint64_t dset, const char *key, int type) {
	if (!tcheck(type)) {
		nonfatal("invalid column data type: %i", type);
		return 0;
	}

	const ds  *d  = handle_lookup(dset, key, 0, 0);
	ds_column *c  = column_lookup(d, key);

	if (!(d && c)) return 0;

	int8_t current_size = abs_i8(type_size[c->type]);
	int8_t proposed_size = abs_i8(type_size[type]);

	if (current_size != proposed_size) {
		nonfatal("cannot change column with type %i to incompatible type %i", c->type, type);
		return 0;
	}

	c->type = type;
	return 1;
}

int dset_addrows (uint64_t dset, uint32_t num) {
	uint64_t idx; 

	ds *d = handle_lookup(dset, "dset_addrows", 0, &idx);
	if (!d) return 0;

	if (d->nrow + num < d->crow) {
		// we already have enough space reserved, so no big deal.
		d->nrow += num;
		return 1;
	}

	// compute the minimum required array heap size for the new row count
	uint64_t req_arrheap_sz = 0;
	for(uint32_t i = 0; i < d->ncol; i++) {

		ds_column *c = d->columns+i;
		req_arrheap_sz += compute_col_reserved_space(d->nrow+num,c);
	}

	// do we have enough space in the heap already?

	uint64_t cur_arrheap_sz = d->strheap_start - d->arrheap_start;
	uint32_t new_crow = d->nrow + num;

	if (req_arrheap_sz > cur_arrheap_sz) {

		// we don't have enough, so we need to make space.
		// but, let's make space for some round number of rows, more than we need.
		uint32_t num_reserve = roundup(num, 100);
		new_crow = d->nrow + num_reserve;

		// we of course need to recompute the required space...
		req_arrheap_sz = 0;
		for(uint32_t i = 0; i < d->ncol; i++) {

			ds_column *c = d->columns+i;
			req_arrheap_sz += compute_col_reserved_space(d->nrow+num_reserve,c);
		}

		d = more_arrheap(idx, req_arrheap_sz-cur_arrheap_sz);
		if(!d) return 0;
	}


	// now we have enough space, we just need to reassign the offsets and do some memmoves
	reassign_arrayoffsets(d, new_crow);

	d->crow  = new_crow;
	d->nrow += num;
	return 1;
}



int dset_defrag (uint64_t dset, int realloc_smaller)
{
	ds *d = handle_lookup(dset, "dset_compress", 0, 0);
	if(!d) return 0;
	char * pd = (char *) d;

	if (d->ccol > d->ncol) {

		char * end = pd + d->strheap_start + d->strheap_sz;

		char * arrheap = pd + d->arrheap_start;
		memmove(d->columns + d->ncol,  arrheap,  end-arrheap);
		d->arrheap_start -= (end-arrheap);
		d->ccol = d->ncol;
	}

	if (d->crow > d->nrow) {
		reassign_arrayoffsets(d, d->nrow);
		d->crow = d->nrow;
	}

	uint64_t actual_heapsz = actual_arrheap_sz(d);
	uint64_t gap = (d->strheap_start - d->arrheap_start) - actual_heapsz;
	if (gap) {
		memmove(pd + d->strheap_start, pd + d->strheap_start - gap, d->strheap_sz);
		d->strheap_start -= gap;
	}

	if (realloc_smaller) {
		d->stats.nrealloc++;
		d = DSREALLOC(d, d->strheap_start + d->strheap_sz);
		if (!d) return 0;
	}

	return 1;
}



const char *dset_getstr (uint64_t dset, const char * colkey, uint64_t index) 
{
	ds        *d = handle_lookup(dset, colkey, 0, 0);
	ds_column *c = column_lookup(d, colkey);

	if(!(d && c)) return 0;

	if (abs_i8(c->type) != T_STR) {
		nonfatal("dset_getstr: column '%s' is not a string", colkey);
		return 0;
	} else {
		return getstr(d, c, index);
	}
}


int dset_setstr (uint64_t dset, const char * colkey, uint64_t index, const char * value)
{
	uint64_t idx;

	ds        *d = handle_lookup(dset, colkey, 0, &idx);
	ds_column *c = column_lookup(d, colkey);

	if(!(d && c)) return 0;

	if (index > d->nrow) {
		nonfatal("dset_setstr: invalid index %"PRIu64, index);
		return 0;
	}

	if (abs_i8(c->type) != T_STR) {
		nonfatal("dset_setstr: column '%s' is not a string", colkey);
		return 0;
	}

	return setstr(d, c, index, value) ? 1 : 0;
}

static char* 
repr_cfloat (uint64_t ds, int sz, char * buf, ds_float_complex_t fc)
{
	snprintf(buf,sz,"(%f,%f)", crealf(fc), cimagf(fc));
	return buf;
}

static char*
repr_cdouble (uint64_t ds, int sz, char * buf, ds_double_complex_t dc)
{
	snprintf(buf,sz,"(%f,%f)", creal(dc), cimag(dc));
	return buf;
}

static char*
repr_str (uint64_t dset, int sz, char * buf, uint64_t handle)
{
	ds     *d = handle_lookup(dset, "repr_str", 0, 0);
	char *ptr = (char *) d;
	snprintf(buf,sz,"%s",ptr + d->strheap_start + handle );
	return buf;
}


void dset_dumptxt (uint64_t dset) {

	ds *d = handle_lookup(dset, "dset_dumptxt", 0, 0);
	xassert(d);


	printf ("dataset %"PRIu64"\n"
		"\ttotal size:            %"PRIu64"\n"
		"\trows (actual)          %"PRIu64"\n"
		"\trows (capacity)        %"PRIu64"\n"
		"\tcols (actual)          %"PRIu32"\n"
		"\tcols (capacity)        %"PRIu32"\n\n"
		"\tnrealloc:              %"PRIu32"\n" 
		"\tnreassign_arroffsets:  %"PRIu32"\n" 
		"\tnshift_strhandles:     %"PRIu32"\n" 
		"\tnmore_arrheap:         %"PRIu32"\n" 
		"\tnmore_strheap:         %"PRIu32"\n" 
		"\tnmore_colspace:        %"PRIu32"\n" ,
		d,
		d->total_sz,
		d->nrow, d->crow,
		d->ncol, d->ccol,
		d->stats.nrealloc,
		d->stats.nreassign_arroffsets,
		d->stats.nshift_strhandles,
		d->stats.nmore_arrheap,
		d->stats.nmore_strheap,
		d->stats.nmore_colspace
		);

	char *sep = "";
	ds_column *c = d->columns;
	for (unsigned i = 0; i < d->ncol; i++,c++) {

		printf("%s%s", sep, getkey(d,c));
		sep = "\t";
	}
	fputc('\n',stdout);

	for (unsigned j = 0; j < d->nrow; j++) {
		c = d->columns;
		for (unsigned i = 0; i < d->ncol; i++,c++) {
			/* TODO/bug: doesn't print non-scalar columns correctly yet */

			char buf[1000];

			char * data = d;
			data += d->arrheap_start + c->offset;

			#define REPR(sym,_a,type,_c,spec,reprfn) \
				case sym: printf("%s" spec, sep, reprfn(dset, sizeof(buf), buf, ((type*)data)[j])); break;
				
			switch (abs_i8(c->type)) {
				DSET_TYPELIST(REPR);
				default:
					fatal("invalid data type");
			}
			sep = "  ";
		}
		fputc('\n',stdout);
	}

}

void *dset_dump(uint64_t dset) {
	return (void *) handle_lookup(dset, "dset_dump", 0, 0);
}

#endif

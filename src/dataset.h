#ifndef DATASET_H
#define DATASET_H

#include <complex.h>   // complex number support
#include <stdint.h>    // fixed width integer types
#include <inttypes.h>  // printf specifiers for fixed width integer types
#include <stdbool.h>   

#define repr(x,a,b,val) val 

#define TYPELIST(X) \
	X(T_F32,  f,   float,           "f4",   "%g", repr ) \
	X(T_F64,  d,   double,          "f8",   "%g", repr ) \
	X(T_C32,  cf,  float complex,   "c8",   "%s", repr_cfloat ) \
	X(T_C64,  cd,  double complex,  "c16",  "%s", repr_cdouble) \
	X(T_I8,   i8,  int8_t,          "i1",   "%" PRIi8,  repr ) \
	X(T_I16,  i16, int16_t,         "i2",   "%" PRIi16, repr ) \
	X(T_I32,  i32, int32_t,         "i4",   "%" PRIi32, repr ) \
	X(T_I64,  i64, int64_t,         "i8",   "%" PRIi64, repr ) \
	X(T_U8,   u8,  uint8_t,         "u1",   "%" PRIu8,  repr ) \
	X(T_U16,  u16, uint16_t,        "u2",   "%" PRIu16, repr ) \
	X(T_U32,  u32, uint32_t,        "u4",   "%" PRIu32, repr ) \
	X(T_U64,  u64, uint64_t,        "u8",   "%" PRIu64, repr ) \
	X(T_STR,  s,   uint64_t,        "O",    "%s", repr_str ) 


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
};

/*
	Not too important for building a python module, but if used in a C program, 
	the user may with to prefix something (e.g. `static` or `__declspec(dllexport)`
	depending on the platform and the user's intent).

	This can be achived by defining DSET_API to something before including this header.
*/
#ifndef DSET_API
#define DSET_API
#endif

DSET_API  uint64_t  dset_new (void);
DSET_API  void      dset_del (uint64_t dset);
DSET_API  uint64_t  dset_copy (uint64_t dset);

DSET_API  uint64_t    dset_totalsz(uint64_t dset);
DSET_API  uint32_t    dset_ncol   (uint64_t dset);
DSET_API  uint64_t    dset_nrow   (uint64_t dset);
DSET_API  int8_t      dset_type   (uint64_t dset, const char * colkey);
DSET_API  void *      dset_get    (uint64_t dset, const char * colkey);
DSET_API  bool        dset_setstr (uint64_t dset, const char * colkey, uint64_t index, const char * value);
DSET_API  const char* dset_getstr (uint64_t dset, const char * colkey, uint64_t index);

DSET_API  bool        dset_addrows       (uint64_t dset, uint32_t num);
DSET_API  bool        dset_addcol_scalar (uint64_t dset, const char * key, int type);
DSET_API  bool        dset_addcol_array  (uint64_t dset, const char * key, int type, const uint8_t shape[3]);


DSET_API  bool        dset_defrag (uint64_t dset, int realloc_smaller);

DSET_API  void        dset_dumptxt (uint64_t dset);


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
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <stdalign.h>
#include <stdarg.h>    // functions with variable number of arguments (e.g. error message callback)

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
	char buf[1024]  = {};
	char buf2[128]  = {};
	char buf3[1024] = {};

	int e = errno;
	if (e != 0) snprintf(buf2,sizeof(buf2)," (errno %d: %s)", e, strerror(e));

	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	snprintf(buf3, sizeof(buf3), "%s%s\n", buf, buf2);
	DSPRINTERR(buf3);
}

static _Noreturn void 
#if defined(__clang__) || defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif
fatal(char *fmt, ...)
{
	char buf[1024]  = {};
	char buf2[128]  = {};
	char buf3[1024] = {};

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
#define xassert(cond) do{(void)(cond)}while(0);
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
	uint32_t   ccol;  // reserved capacity for columns
	uint32_t   ncol;  // actual number of columns
	uint64_t   crow;  // reserved capacity for rows
	uint64_t   nrow;  // actual number of rows
	uint64_t   total_sz; 
	uint64_t   arrheap_start;
	uint64_t   strheap_start;
	uint64_t   strheap_sz;
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

	pthread_once_t    init_guard;
	pthread_mutex_t   mtx;

	uint64_t          nslots; 
	ds_slot *         slots;

} ds_module = {
	.init_guard = PTHREAD_ONCE_INIT,
};

static void
_module_init(void) 
{
	// initialize the mutex, and enable some protections against programmer errors
	pthread_mutexattr_t a;
	xassert(0 == pthread_mutexattr_init(&a));
	xassert(0 == pthread_mutexattr_settype(&a, PTHREAD_MUTEX_ERRORCHECK));

	xassert(0 == pthread_mutex_init(&ds_module.mtx, &a));

}

static inline void
module_init(void) {
/*
	make sure that we can call module_init() as often as we want but the underlying init
	happens exactly once.
*/

	xassert(0 == pthread_once(&ds_module.init_guard, _module_init));
}

static inline void
lock (void) {
/*
	This lock only needs to be held when creating or destroying datasets.
	We don't guarantee that datasets can be safely accessed concurrently, that's up to the user.
*/
	int rc = pthread_mutex_lock(&ds_module.mtx);
	errno = rc;
	xassert(rc == 0);
}

static inline void
unlock (void) {
	int rc = pthread_mutex_unlock(&ds_module.mtx);
	errno = rc;
	xassert(rc == 0);
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

#define SHIFT_GEN (64-15)
#define MASK_IDX  (0xffffffffffffffff >> SHIFT_GEN)

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

	// see if we can find an existing empty slot
	uint64_t i = 0;
	for (i = 0; i < ds_module.nslots; i++) {

		if (! ds_module.slots[i].memory) break;
	}

	if (i == ds_module.nslots)
		moreslots();

	if (i == ds_module.nslots) 
		goto out_of_memory;


	ds_slot *s = &ds_module.slots[i];

	void * mem = DSREALLOC(0, newsize);
	if (!mem) goto out_of_memory;

	*allocation = mem;
	s->memory   = mem;
	unlock();

	memset(s->memory, 0, newsize);
	uint64_t gen = ++s->generation;

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
		nonfatal("%s: invalid handle %" PRIx64 ", no such slot", msg_fragment, h);
		return 0;
	}

	if (!ds_module.slots[*idx].memory) { 
		nonfatal("%s: invalid handle %" PRIx64 ", no heap at index %" PRIu64, msg_fragment, h, *idx);
		return 0;
	}

	if (ds_module.slots[*idx].generation != *gen) {
		nonfatal("%s: invalid handle %" PRIx64 ", wrong generation counter"
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
int valid_types[] = { TYPELIST(EMIT_TYPEENUM_ONLY) };

static const 
size_t Ntypes = sizeof(valid_types)/sizeof(valid_types[0]);

#define EMIT_SZ_ARRAY_ENTRY(typeenum,a,ctype,b,c,d) [typeenum]=sizeof(ctype),
static const 
size_t type_size[] = { TYPELIST(EMIT_SZ_ARRAY_ENTRY) };

#define EMIT_ALIGNCHECK(a,name,ctype,b,c,d) static_assert(alignof(ctype) <= 16, "platform incompatible");
TYPELIST(EMIT_ALIGNCHECK) ;


static int8_t abs_i8 (int8_t x) {return x < 0 ? -x : x;}
#define EMIT_TYPECHECK_FUNCTION(typeenum, fnsuffix, a,b,c,d) \
	static bool CONCAT(tcheck_, fnsuffix) (int8_t type) {     \
		return abs_i8(type) == typeenum;    \
	}
TYPELIST(EMIT_TYPECHECK_FUNCTION)

static bool
tcheck(int8_t type)
{
	const int t  = abs_i8(type);
	
	for (int i = 0; i < Ntypes; i++) 
		if (t == valid_types[i]) 
			return true;

	return false;
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

	nonfatal("key error: %s", colkey);
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

	const uint64_t more = roundup(nbytes_more, 1<<25); // 32 MB at a time (too little? too much?)

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

		memmove (move_dst, move_src, nbytes_more);
		memset  (move_dst + d->strheap_sz, 0, nbytes_more);

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

			memmove (move_dst, move_src, nbytes_more);
			memset  (move_src, 0,        nbytes_more);

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

			memmove (move_dst, move_src, nbytes_more);
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
	if (!oldstr) return;
	char * strheap = ((char *)d) + d->strheap_start;
	char * s = strheap + oldstr;
	int64_t sz = 1 + strlen(s);
	memmove(s, s+sz, (strheap+d->strheap_sz) - (s+sz));
	shift_all_string_handles(d, -sz, oldstr);
	d->strheap_sz -= sz;
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


/*
===============================================================================
                           ACTUAL API FUNCTIONS
===============================================================================
*/



DSET_API uint64_t 
dset_new(void) {
	const size_t DS_INITIAL_SZ = 1<<25; // 32MB as a good default?
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

	return handle;
}

DSET_API uint64_t
dset_copy(uint64_t dset)
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

DSET_API void
dset_del(uint64_t dset)
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

DSET_API uint64_t 
dset_totalsz(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_ncol", 0, 0);
	if(d) return d->total_sz;
	else  return 0;
}

DSET_API uint32_t 
dset_ncol(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_ncol", 0, 0);
	if(d) return d->ncol;
	else  return 0;
}

DSET_API uint64_t 
dset_nrow(uint64_t dset)
{
	ds *d = handle_lookup(dset, "dset_nrow", 0, 0);
	if(d) return d->nrow;
	else  return 0;
}

DSET_API  int8_t
dset_type (uint64_t dset, const char * colkey)
{
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);

	if(!(d && c)) return 0;
	return abs_i8(c->type);
}

DSET_API void *    
dset_get (uint64_t dset, const char * colkey)
{
	const ds        *d  = handle_lookup(dset, colkey, 0, 0);
	const ds_column *c  = column_lookup(d, colkey);
	char * ptr = (char *) d;

	if(!(d && c)) return 0;

	if (abs_i8(c->type) == T_STR) {
		nonfatal("dset_get: column '%s' is a string", colkey);
		return 0;
	}

	return ptr + d->arrheap_start + c->offset;
}


DSET_API bool 
dset_addcol_scalar (uint64_t dset, const char * key, int type) {
	const uint8_t shape[3] = {};
	return dset_addcol_array(dset, key, type, shape);
}


DSET_API bool 
dset_addcol_array (uint64_t dset, const char * key, int type, const uint8_t shape[3]) {

	if(!tcheck(type)) {
		nonfatal("invalid column data type: %i", type);
		return false;
	}

	uint64_t idx; 
	ds *d = handle_lookup(dset, "add column", 0, &idx);
	if(!d) return false;

	const size_t ksz = 1 + strlen(key);
	const int8_t t   = abs_i8(type);

	// hypothetical new column descriptor.
	ds_column col = {};
	col.type      =  ksz > SHORTKEYSZ ? -t : t;
	memcpy(col.shape,shape,sizeof(col.shape));

	if (d->ncol == d->ccol) {

		// need more space for column descriptors
		d = more_columndescr_space(idx, 30);
		if (!d) return false;
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
		if (!d) return false;
	}

	if (col.type < 0) {
		// key is long, so allocate a string for it	

		uint64_t newstr = stralloc(&d, idx, key);
		if (!d) return false;
		col.longkey    = newstr;

	} else {
		snprintf(col.shortkey, sizeof(col.shortkey), "%s", key);
	}

	// commit the new column
	d->columns[d->ncol++] = col;
	return true;
}


DSET_API bool 
dset_addrows (uint64_t dset, uint32_t num) {
	uint64_t idx; 

	ds *d = handle_lookup(dset, "dset_addrows", 0, &idx);
	if (!d) return false;

	if (d->nrow + num < d->crow) {
		// we already have enough space reserved, so no big deal.
		d->nrow += num;
		return true;
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
		if(!d) return false;
	}


	// now we have enough space, we just need to reassign the offsets and do some memmoves
	reassign_arrayoffsets(d, new_crow);

	d->crow  = new_crow;
	d->nrow += num;
	return true;
}



DSET_API bool 
dset_defrag (uint64_t dset, int realloc_smaller)
{
	ds *d = handle_lookup(dset, "dset_compress", 0, 0);
	if(!d) return false;
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
		if (!d) return false;
	}

	return true;
}



DSET_API const char *    
dset_getstr (uint64_t dset, const char * colkey, uint64_t index) 
{
	ds        *d = handle_lookup(dset, colkey, 0, 0);
	ds_column *c = column_lookup(d, colkey);
	char * ptr = (char *) d;

	if(!(d && c)) return 0;

	if (abs_i8(c->type) != T_STR) {
		nonfatal("dset_getstr: column '%s' is not a string", colkey);
		return 0;
	} else {
		uint64_t *handles = ptr + d->arrheap_start + c->offset;
		return ptr + d->strheap_start + handles[index];
	}
}


DSET_API bool
dset_setstr (uint64_t dset, const char * colkey, uint64_t index, const char * value)
{
	uint64_t idx;

	ds        *d = handle_lookup(dset, colkey, 0, &idx);
	ds_column *c = column_lookup(d, colkey);
	char * ptr = (char *) d;

	if(!(d && c)) return 0;

	if (index > d->nrow) {
		nonfatal("dset_setstr: invalid index %"PRIu64, index);
		return 0;
	}

	if (abs_i8(c->type) != T_STR) {
		nonfatal("dset_setstr: column '%s' is not a string", colkey);
		return 0;
	}

	uint64_t *handles = ptr + d->arrheap_start + c->offset;

	strfree(handles[index], d);		
	uint64_t newstr = stralloc(&d,idx,value);
	if(!d) return 0;

	handles[index] = newstr;
	return true;
}


static char* 
repr_cfloat (uint64_t ds, int sz, char * buf, float complex fc)
{
	snprintf(buf,sz,"(%f,%f)", crealf(fc), cimagf(fc));
	return buf;
}

static char*
repr_cdouble (uint64_t ds, int sz, char * buf, double complex dc)
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


DSET_API  void 
dset_dumptxt (uint64_t dset) {

	ds *d = handle_lookup(dset, "dset_dumptxt", 0, 0);
	xassert(d);


	printf ("dataset %"PRIx64"\n"
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

			char buf[1000] = {};

			char * data = d;
			data += d->arrheap_start + c->offset;

			#define REPR(sym,_a,type,_c,spec,reprfn) \
				case sym: printf("%s" spec, sep, reprfn(dset, sizeof(buf), buf, ((type*)data)[j])); break;
				
			switch (abs_i8(c->type)) {
				TYPELIST(REPR);
				default:
					fatal("invalid data type");
			}
			sep = "  ";
		}
		fputc('\n',stdout);
	}

}

#endif

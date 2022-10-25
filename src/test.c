#include <stdio.h>

#define DATASET_IMPLEMENTATION
#include "dataset.h"
#include <math.h>

char * randstr(void) {
	static char str[10];
	const int modulus = 'z'-'0';
	for (int i = 0; i < 9; i++)
		str[i] = rand() % modulus + '0';
	return str;
}

#include <time.h>
int main (int argc, char ** argv) 
{
	srand(time(NULL));
	uint64_t d = dset_new();

	printf("ncol %u \n", dset_ncol(d));
	printf("nrow %u \n", dset_nrow(d));

	xassert(dset_addrows(d, 10));
	xassert(dset_addcol_scalar(d, "col1", T_F32));
	xassert(dset_addcol_scalar(d, "col2", T_F32));
	xassert(dset_addcol_scalar(d, "col3", T_F32));

	float * col1 = dset_get(d, "col1");
	float * col2 = dset_get(d, "col2");
	float * col3 = dset_get(d, "col3");

	#define LONGSTR "veryveryveryveryveryveryveryveryveryveryveryveryveryveryveryveryvery long string as column header"
	xassert(dset_addcol_scalar(d, LONGSTR, T_STR));

	for (int i = 0; i < 10; i++) {
		col1[i] = sinf((float)i);
		col2[i] = cosf((float)i);
		col3[i] = tanf((float)i);
		xassert(dset_setstr(d, LONGSTR, i, randstr()));
	}


	printf("ncol %u \n", dset_ncol(d));
	printf("nrow %u \n", dset_nrow(d));
	printf("\n");
	dset_dumptxt(d);
	printf("\n");

	for (int i = 0; i < 10; i++) {
		xassert(dset_setstr(d, LONGSTR, i, randstr()));
	}
	dset_dumptxt(d);
	printf("\n");

	dset_del(d);

	// Hashtable tests
	uint64_t val;
	ht64 ht;
	ht64_new(&ht, 3);
	xassert(ht.len == 0);
	xassert(ht.exp == 2);

	// Find on empty
	xassert(!ht64_find(&ht, 42, 0));

	// Insert and find
	xassert(ht64_insert(&ht, 42, 123));
	xassert(ht64_find(&ht, 42, &val));
	xassert(val == 123);

	// Fill up table
	xassert(ht64_insert(&ht, 43, 456));
	xassert(ht64_insert(&ht, 44, 789));
	xassert(ht64_insert(&ht, 45, 12345));
	xassert(!ht64_insert(&ht, 46, 56789)); // full

	// Deallocate
	ht64_del(&ht);
}

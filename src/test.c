#include <stdio.h>

#define DATASET_IMPLEMENTATION
#include "dataset.h"
#include <math.h>

int main (int argc, char ** argv) 
{
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

	for (int i = 0; i < 10; i++) {
		col1[i] = sinf((float)i);
		col2[i] = cosf((float)i);
		col3[i] = tanf((float)i);
	}


	printf("ncol %u \n", dset_ncol(d));
	printf("nrow %u \n", dset_nrow(d));

	dset_dumptxt(d);

	dset_del(d);
}

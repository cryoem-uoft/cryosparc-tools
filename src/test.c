#include <stdio.h>

#define DATASET_IMPLEMENTATION
#include "dataset.h"

int main (int argc, char ** argv) 
{
	uint64_t d = dset_new();

	dset_del(d);
}

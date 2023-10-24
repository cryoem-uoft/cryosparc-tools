#!/usr/bin/env sh
# For testing native module compilation
set -e
cc -g test.c -I ../cryosparc/include -lpthread -lm
./a.out | column -t

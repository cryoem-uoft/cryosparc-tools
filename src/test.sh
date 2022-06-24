#!/usr/bin/env sh
set -e
cc -g test.c -lpthread -lm
./a.out | column -t


#!/usr/bin/env sh
set -e
PY_INCLUDES=" -I$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])') -I$(python -c 'import numpy; print(numpy.get_include())') "
PY_LIBDIR="-L/usr/local/lib"
cc -fPIC -lpthread -lm -DMODULENAME=dataset -g $PY_INCLUDES $PY_LIBDIR pywrapper_module.c pywrapper_dataset.c -lpython3.8 -shared -o dataset.so

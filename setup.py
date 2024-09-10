import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

DEBUG = False  # set to True to enable debugging
libraries = []
define_macros = []
undef_macros = []
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args += ["/std:c11"]
    define_macros += [("__WIN32__", 1)]
else:
    libraries.append("pthread")

if sys.platform == "win32" and DEBUG:
    define_macros += [("_DEBUG",)]
    extra_compile_args += ["/Zi"]
    extra_link_args += ["/DEBUG"]
elif DEBUG:
    extra_compile_args += ["-g", "-O0", "-Wall", "-Wextra"]

setup(
    name="cryosparc_tools",
    version="4.6.0",
    description="Toolkit for interfacing with CryoSPARC",
    headers=["cryosparc/include/cryosparc-tools/dataset.h"],
    ext_modules=cythonize(
        Extension(
            name="cryosparc.core",
            sources=[
                "cryosparc/include/lz4/lib/lz4.c",
                "cryosparc/dataset.c",
                "cryosparc/core.pyx",
            ],
            include_dirs=["cryosparc/include/"],
            libraries=libraries,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            depends=[
                "cryosparc/include/lz4/lib/lz4.h",
                "cryosparc/include/cryosparc-tools/dataset.h",
                "cryosparc/lz4.pxd",
                "cryosparc/dataset.pxd",
            ],
        ),
        language_level=3,
        gdb_debug=DEBUG,
    ),
)

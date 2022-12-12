import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

DEBUG = False  # set to True to enable debugging
libraries = []
define_macros = []
undef_macros = []
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args += ["/std:c11"]
else:
    libraries.append("pthread")

if sys.platform == "win32" and DEBUG:
    define_macros += [("_DEBUG",)]
    extra_compile_args += ["/Zi"]
    extra_link_args += ["/DEBUG"]
elif DEBUG:
    extra_compile_args += ["-g", "-O0"]

setup(
    name="cryosparc_tools",
    version="4.1.0",
    description="Toolkit for interfacing with CryoSPARC",
    headers=["cryosparc/include/cryosparc-tools/dataset.h"],
    ext_modules=cythonize(
        Extension(
            name="cryosparc.core",
            sources=["cryosparc/dataset.c", "cryosparc/core.pyx"],
            include_dirs=["cryosparc/include/"],
            libraries=libraries,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            depends=["cryosparc/include/cryosparc-tools/dataset.h", "cryosparc/dataset.pxd"],
        ),
        language_level=3,
        gdb_debug=DEBUG,
    ),
)

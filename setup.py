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

if DEBUG:
    define_macros += [("NDEBUG",)]
else:
    undef_macros += ["NDEBUG"]

if sys.platform == "win32" and DEBUG:
    define_macros += [("_DEBUG",)]
    extra_compile_args += ["/Zi"]
    extra_link_args += ["/DEBUG"]
elif DEBUG:
    extra_compile_args += ["-g", "-O0"]

setup(
    name="cryosparc_tools",
    version="4.0.0",
    description="Toolkit for interfacing with CryoSPARC",
    headers=["src/dataset.h"],
    package_data={"cryosparc": ["src/dataset.h", "cryosparc/core.pyx"]},
    ext_modules=cythonize(
        Extension(
            name="cryosparc.core",
            sources=["src/pywrapper_dataset.c", "cryosparc/core.pyx"],
            include_dirs=["src/"],
            libraries=libraries,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            depends=["src/dataset.h"],
        ),
        gdb_debug=DEBUG
    ),
)

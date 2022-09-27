import sys
from setuptools import Extension, setup

DEBUG = False  # set to True to enable debugging
libraries = []
define_macros = [("MODULENAME", "core")]
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
    version="0.1.0",
    description="Toolkit for interfacing with CryoSPARC",
    headers=["src/dataset.h"],
    ext_modules=[
        Extension(
            name="cryosparc.core",
            libraries=libraries,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            sources=["src/pywrapper_dataset.c", "src/pywrapper_module.c"],
            depends=["src/dataset.h", "src/pywrapper_wrapperfunctions.c", "src/pywrapper_extras.c"],
        )
    ],
)

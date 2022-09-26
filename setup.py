import sys
from setuptools import Extension, setup

libraries = ["m"]
if sys.platform != "win32":
    libraries.append("pthread")

setup(
    name="cryosparc_tools",
    version="0.1.0",
    description="Toolkit for interfacing with CryoSPARC",
    headers=["src/dataset.h"],
    ext_modules=[
        Extension(
            name="cryosparc.core",
            libraries=libraries,
            define_macros=[("MODULENAME", "core")],
            sources=["src/pywrapper_dataset.c", "src/pywrapper_module.c"],
            depends=["src/dataset.h", "src/pywrapper_wrapperfunctions.c", "src/pywrapper_extras.c"],
        )
    ],
)

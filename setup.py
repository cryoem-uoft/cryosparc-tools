import numpy
from setuptools import Extension, setup


setup(
    name="cryosparc_tools",
    version="0.1.0",
    description="Toolkit for interfacing with cryoSPARC",
    headers=["src/dataset.h"],
    ext_modules=[
        Extension(
            name="cryosparc.core",
            libraries=["pthread", "m"],
            include_dirs=[numpy.get_include()],
            define_macros=[("MODULENAME", "core")],
            sources=["src/pywrapper_dataset.c", "src/pywrapper_module.c"],
            depends=["src/dataset.h", "src/pywrapper_wrapperfunctions.c", "src/pywrapper_extras.c"],
        )
    ],
)

import numpy
from setuptools import Extension, setup


setup(
    name="cryosparc_tools",
    version="0.1.0",
    description="Toolkit for interfacing with cryoSPARC",
    headers=["src/dataset.h"],
    ext_modules=[
        Extension(
            name="cryosparc_dataset",
            libraries=['pthread', 'm'],
            include_dirs=[numpy.get_include()],
            define_macros=[('MODULENAME', 'cryosparc_dataset')],
            sources=["src/pywrapper_module.c", "src/pywrapper_dataset.c"],
            depends=["src/dataset.h", "src/pywrapper_wrapperfunctions.c"],
        )
    ],
)

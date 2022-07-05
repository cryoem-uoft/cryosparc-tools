import os
import sysconfig
import numpy
from setuptools import Extension, setup


setup(
    name="cryosparc_tools",
    version="0.1.0",
    description="Toolkit for interfacing with cryoSPARC",
    ext_modules=[
        Extension(
            name="cryosparc_dataset",
            sources=["src/pywrapper_module.c", "src/pywrapper_dataset.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-DMODULENAME=cryosparc_dataset"],
            extra_link_args=["-lpthread", "-lm"],
        )
    ],
)

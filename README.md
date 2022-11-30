# cryosparc-tools

Toolkit for interfacing with CryoSPARC. Read the documentation at
[tools.cryosparc.com](https://tools.cryosparc.com)

## Development

### Prerequisites

- Git and Git LFS
- Python >= 3.7
- Miniconda3
- C compiler such as GCC or Clang

### Set Up

1. Clone this repository
   ```sh
   git clone https://github.com/cryoem-uoft/cryosparc-tools.git
   cd cryosparc-tools
   git lfs pull
   ```
2. Create and activate a conda environment named "tools" with the desired python version. See the Run Example Notebooks section to install an environment
   ```sh
   conda create -n tools python=3.7 -c conda-forge
   conda activate tools
   ```
3. Install dev dependencies and build native modules
   ```sh
   pip install -U pip wheel
   pip install -e ".[dev]"
   ```

### Re-compile native module

Recompile native modules after making changes to C code:

```sh
make
```

## Build Packages for Publishing

Install build dependencies

```sh
pip install -e ".[build]"
```

Run the build

```sh
python -m build
```

Packages for the current architecture and python version are added to the
`dist/` directory.

## Build Documentation

Documentation is located in the `docs` directory and is powered by [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

To build the docs, install build dependencies

```sh
pip install -e ".[build]"
```

Then run Jupyter Book

```sh
jupyter-book build docs
```

Site will be be built into the `docs/_build/html` directory.

**Note:** Jupyter Book is not configured to re-run example notebooks upon build
since the notebooks require an active CryoSPARC instance to run.

See the [Run Example Notebooks](#run-example-notebooks) section for instructions
on how to run the notebooks.

Inline source documentation is compiled to HTML via [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and uses [Google Style Python docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google).

## Run Example Notebooks

The Jupyter notebooks in the example documentation require additional
dependencies to execute, including the following system configuration:

- Nvidia GPU and driver
- CryoSPARC running at `localhost:39000`

Clean previous build artefacts:

```sh
make clean
```

Install additional dependencies to conda environment before running `pip`:

```sh
conda create -n tools -c conda-forge \
   python=3.8 \
   cudatoolkit=11.6 \
   cudnn=8.3 \
   libtiff \
   notebook \
   pyqt=5 \
   wxPython=4.1.1 \
   adwaita-icon-theme
conda activate tools
```

Install notebook dependencies with `pip`.

```sh
pip install -U pip
pip install nvidia-pyindex
```

Install example deps and rebuild

```sh
pip install -e ".[examples]"
make
```

Run the notebook server with the `CRYOSPARC_LICENSE_ID` environment variable
containing a CryoSPARC License, open in the browser. You may also need to
include `LD_LIBRARY_PATH` which includes the location of CUDA Toolkit and cuDNN
runtime libraries (e.g., `~/miniconda3/envs/tools/lib`).

```
CRYOSPARC_LICENSE_ID="xxxxxxxx-xxxx-xxxx-xxxxxxxxxxxx" jupyter notebook
```

Find examples in `docs/examples` directory

## License

cryosparc-tools is licensed under the BSD-3-Clause.

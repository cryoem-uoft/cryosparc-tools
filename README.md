# cryosparc-tools

[![PyPI](https://badgen.net/pypi/v/cryosparc-tools)](https://pypi.org/project/cryosparc-tools/)
[![Python package](https://github.com/cryoem-uoft/cryosparc-tools/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/cryoem-uoft/cryosparc-tools/actions/workflows/python-package.yml)

Toolkit for interfacing with CryoSPARC. Read the documentation at
[tools.cryosparc.com](https://tools.cryosparc.com)

## Getting Help and Reporting Bugs

For usage questions and general discussion about `cryosparc-tools` scripts and functions, please post to the [CryoSPARC discussion forum](https://discuss.cryosparc.com/c/scripting) under the Scripting category.

If you would like to request or propose a feature, change or fix for `cryosparc-tools` source code, please either [report an issue](https://github.com/cryoem-uoft/cryosparc-tools/issues/new) or [submit a pull request](https://github.com/cryoem-uoft/cryosparc-tools/compare).

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Development

### Prerequisites

- Git and Git LFS
- Python >= 3.7
- Miniconda3
- C compiler such as GCC or Clang

### Set Up

1. Clone this repository
   ```sh
   git clone --recursive https://github.com/cryoem-uoft/cryosparc-tools.git
   cd cryosparc-tools
   git lfs pull
   ```
2. (Optional) Create and activate a virtual environment
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   # OR
   .venv\Scripts\activate.bat  # Windows
   ```
3. Install dev dependencies and build native modules
   ```sh
   pip install -U pip wheel
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks
   ```
   pre-commit install
   ```

### Re-compile native module

Recompile native modules after making changes to C code:

```sh
python -m setup build_ext -i
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
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- CryoSPARC running at `localhost:40000` or `cryoem0:40000` (can alias `cryoem0` to localhost)

Clean previous build artefacts:

```sh
rm -rf cryosparc/*.so build dist *.egg-info
```

Install dependencies into a new conda environment:

```sh
conda create -n cryosparc-tools-example -c conda-forge \
    python=3 numpy==1.18.5 \
    pyqt=5 libtiff wxPython=4.1.1 adwaita-icon-theme
conda activate cryosparc-tools-example
pip install -U pip
pip install nvidia-pyindex matplotlib~=3.4.0 pandas==1.1.4 notebook
pip install "cryolo[c11]"
pip install -e ".[build]"
```

Run the notebook server with the following environment variables:

- `CRYOSPARC_LICENSE_ID` with Structura-issued CryoSPARC license
- `CRYOSPARC_EMAIL` with a CryoSPARC user account email
- `CRYOSPARC_PASSWORD` with a CryoSPARC user account password

You may also need to include `LD_LIBRARY_PATH` which includes the location of
CUDA Toolkit and cuDNN runtime libraries (e.g., `~/miniconda3/envs/tools/lib/python3.8/site-packages/nvidia/*/lib`).

```
CRYOSPARC_LICENSE_ID="xxxxxxxx-xxxx-xxxx-xxxxxxxxxxxx" \
CRYOSPARC_EMAIL="ali@example.com" \
CRYOSPARC_PASSWORD="password123" \
jupyter notebook
```

Find examples in `docs/examples` directory

## License

cryosparc-tools is licensed under the BSD-3-Clause license.

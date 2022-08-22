cryosparc-tools
===

Toolkit for interfacing with cryoSPARC

## Development

### Prerequisites

* Git LFS
* Python >= 3.7
* Miniconda3
* C compiler such as GCC or Clang

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
   pip install -U pip
   pip install -e ".[dev]"
   ```

### Re-compile native module

Recompile native modules after making changes to C code:

```sh
python -m setup develop
```

### Build Packages for Publishing

```
python -m build
```

Packages for the current architecture and python version are added to the
`dist/` directory.


### Run Example Notebooks

The Jupyter notebooks in the example documentation require additional
dependencies to execute, including the following system configuration:

* Nvidia GPU and driver
* cryoSPARC running at `localhost:3000`

Open `pyproject.toml`, look for the `[build-system]` line. Edit the `requires`
to use `numpy >=1.18.0<1.19.0` instead of the normal `numpy ~= 1.15`. Do not
commit this change.

Clean previous build artefacts:

```sh
rm -rf build dist *.egg-info cryosparc/*.so
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

Rebuild

```
pip install -e ".[notebook]"
python -m setup develop
```

Run the notebook server with the `CRYOSPARC_LICENSE_ID` environment variable
containing a cryoSPARC License, open in the browser. You may also need to
include `LD_LIBRARY_PATH` which includes the location of CUDA Toolkit and cuDNN
runtime libraries (e.g., `~/miniconda3/envs/tools/lib`).

```
CRYOSPARC_LICENSE_ID="xxxxxxxx-xxxx-xxxx-xxxxxxxxxxxx" jupyter notebook
```

Find examples in `docs/examples` directory

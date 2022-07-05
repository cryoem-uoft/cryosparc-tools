cryosparc-tools
===

Toolkit for interfacing with cryoSPARC

## Development

### Prerequisites

* Python >= 3.7
* C compiler such as GCC or Clang

### Set Up

1. Clone this repository
   ```sh
   git clone https://github.com/cryoem-uoft/cryosparc-tools.git
   cd cryosparc-tools
   ```
2. Create and activate a virtual environment
   ```sh
   python -m venv .venv
   source .venv/bin/activate
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

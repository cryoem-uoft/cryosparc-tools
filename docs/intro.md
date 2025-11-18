# Getting Started

**cryosparc-tools** is an open-source Python library that enables scripting access to the [CryoSPARC](https://cryosparc.com) <abbr title="Cryogenic-electron microscopy">cryo-EM</abbr> software package.

![CryoSPARC Architecture with cryosparc-tools](_static/cryosparc-tools-architecture.png)

Use it for the following use cases:

- Programmatically read and write exposure, particle and volume data
- Access project, workspace and job data
- Build and run jobs to orchestrate custom cryo-EM workflows
- Extend CryoSPARC functionality with third-party software packages

```{note}
This guide documents usage of the `cryosparc-tools` Python library. For CryoSPARC installation or general CryoSPARC usage instructions, [read the official guide](https://guide.cryosparc.com).
```

For usage questions and general discussion about `cryosparc-tools` scripts and functions, please post to the [CryoSPARC discussion forum](https://discuss.cryosparc.com/c/scripting) under the Scripting category.

If you would like to request or propose a feature, change or fix for `cryosparc-tools` source code, please either [report an issue](https://github.com/cryoem-uoft/cryosparc-tools/issues/new) or [submit a pull request](https://github.com/cryoem-uoft/cryosparc-tools/compare).

Source code is [available on GitHub](https://github.com/cryoem-uoft/cryosparc-tools).

## Pre-requisites

- [Python ≥ 3.8](https://www.python.org/downloads/)
- [CryoSPARC ≥ v4.1](https://cryosparc.com/download)
- A terminal program to run commands

CryoSPARC installation must be accessible via one of the following methods:

- Running on the local machine
- Running on a machine on the same network with `BASE_PORT` open for connections
- Running at some publicly accessible web URL, e.g., `https://cryosparc.example.com`

```{note}
CryoSPARC Tools versions prior to v5 require that CryoSPARC instances accessed over SSH have multiple ports forwarded (`BASE_PORT + 2`, `BASE_PORT + 3`, `BASE_PORT + 5`).

As of v5, only the `BASE_PORT`, e.g., 39000, needs to be forwarded.

See [SSH Port Forwarding documentation](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/accessing-cryosparc#ssh-local-port-forwarding) for more information.
```

cryosparc-tools is only available for CryoSPARC v4.1 or newer. If using CryoSPARC v4.0 or older, please see the [Manipulating .cs Files Created By CryoSPARC](https://guide.cryosparc.com/setup-configuration-and-management/software-system-guides/manipulating-.cs-files-created-by-cryosparc) guide.

### Python Environment

cryosparc-tools is intended to be used in a dedicated Python environment
_outside_ of the CryoSPARC installation. A virtual environment is recommended to
avoid conflicts with global Python installations.

Virtual environment tools such as
[venv](https://docs.python.org/3/tutorial/venv.html),
[Conda](https://docs.conda.io/en/latest/),
[Mamba](https://mamba.readthedocs.io/en/latest/),
[Poetry](https://python-poetry.org) and
[uv](https://docs.astral.sh/uv/) all work with cryosparc-tools.

Ensure that the virtual environment is based on a supported version of Python
(see Pre-requisites above).

## Installation

In a terminal, enter the following command to install the latest version of cryosparc-tools from [PyPI](https://pypi.org) into the current Python environment:

```sh
pip install cryosparc-tools
```

Alternatively, update an existing installation of cryosparc-tools to the latest version:

```sh
pip install -U cryosparc-tools
```

```{note}
Use the version of cryosparc-tools that corresponds to your CryoSPARC _minor_
release version. i.e., if the CryoSPARC version is vX.Y.Z, use the latest vX.Y
tools package. The Z component does not need to match.

For example, if you are running CryoSPARC v4.1.2, install cryosparc-tools with
`pip install cryosparc-tools~=4.1.0` (equivalent to `pip install "cryosparc-tools>=4.1.0,<4.2"`).
If you later update to CryoSPARC v4.2.0 or v5.0.0, re-install the corresponding
tools package with `pip install cryosparc-tools~=4.2.0` or
`pip install cryosparc-tools~=5.0.0` respectively.
```

## Usage

In a terminal, enter the following command to log in to your CryoSPARC instance:

```sh
python -m cryosparc.tools login --url <URL>
```

Replace `<URL>` with the URL you use to access CryoSPARC from your web browser, e.g., `http://localhost:39000`. Enter your CryoSPARC email and password when prompted. This saves a login token to a local file that expires in two weeks.

```{note}
You only need to log in once, unless the token expires or you change your CryoSPARC password.
```

Create a new file in a text editor such as [VS Code](https://code.visualstudio.com), add the following text:

```py
from cryosparc.tools import CryoSPARC

cs = CryoSPARC("<URL>")
assert cs.test_connection()
```

Replace `<URL>` with the same URL specified above.

When run, this script imports the `CryoSPARC` function, calls it and stores the result in the `cs` variable. This variable represents a connection to your CryoSPARC instance.

Save the file with a descriptive name, e.g., `data_processing.py`, and run it in a terminal like this:

```sh
cd /path/to/scripts  # replace with the path to the folder where you saved the script
python data_processing.py
```

You should see the message `Success: Connected to CryoSPARC API at <URL>` printed to the terminal.

cryosparc-tools allows you to query projects, jobs and result datasets. For example, you can add the following code to your script to print the paths of all micrographs in a motion correction job with ID `J42` in project `P3`:

```py
project = cs.find_project("P3")
job = project.find_job("J42")
micrographs = job.load_output("micrographs")

for mic in micrographs.rows():
    print(mic["micrograph_blob/path"])
```

You may also load and save CryoSPARC dataset files. This assumes the project directory is available on the same file system as the script:

```py
from cryosparc.dataset import Dataset

path = project.dir / "J43" / "particles.cs"
particles = Dataset.load(path)

shift_y, shift_x = particles["alignments2D/shift"].T
mic_shape_y, mic_shape_x = particles["location/micrograph_shape"].T
new_loc_x = particles["location/center_x_frac"] * mic_shape_x - shift_x
new_loc_y = particles["location/center_y_frac"] * mic_shape_y - shift_y
particles["location/center_x_frac"] *= new_loc_x / mic_shape_x
particles["location/center_y_frac"] *= new_loc_y / mic_shape_y

particles.save(path)
```

If the project is on a remote machine, you may download the dataset locally first:

```py
particles = project.download_dataset("J43/particles.cs")
shift_y, shift_x = particles["alignments2D/shift"].T
...
project.upload_dataset("J43/particles.cs", particles)
```

Browse the included guides and examples to get a better idea of what you can do with cryosparc-tools.

For full details about available functions and classes in cryosparc-tools, including their capabilities and expected arguments, read the <abbr title="Application Programming Interface">API</abbr> Reference. The [`cryosparc.tools` module](api/cryosparc.tools) is the best place to start.

## Jupyter Notebooks

We recommend writing and using CryoSPARC tools in a [Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/latest/). Jupyter notebooks allow for a more interactive and iterative use of Python.

In a Jupyter notebook, time-consuming steps (like data loading) can run a single time without slowing down later steps (like plotting) which are quick, but depend on those earlier steps.

To install Jupyter:

```sh
pip install notebook
```

Run Jupyter:

```sh
jupyter notebook
```

Note the login token in the output.

This starts a Jupyter Notebook server at http://localhost:8888 on the current
machine. Optionally provide the following arguments to make Jupyter available to
other machines on the local network:

```sh
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

(running-the-examples)=

### Running the Examples

The example Jupyter notebooks require additional dependencies to run. Use [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/) to create a new Python environment with the required dependencies. Here, the environment is named `cryosparc-tools-example` but any name may be used:

```sh
conda create -n cryosparc-tools-example -c conda-forge python=3 numpy==1.18.5
conda activate cryosparc-tools-example
pip install matplotlib~=3.4.0 pandas==1.1.4 cryosparc-tools
```

For speed, these do not include the dependencies for the crYOLO example
notebook. Optionally install crYOLO with these commands:

```sh
conda install -c conda-forge pyqt=5 libtiff wxPython=4.1.1 adwaita-icon-theme 'setuptools<66'
pip install 'cryolo[c11]' --extra-index-url https://pypi.ngc.nvidia.com
```

Then proceed with the Jupyter notebook installation above.

Example notebooks ran on Ubuntu Linux with x86-64 bit architecture.

## Next Steps

```{tableofcontents}

```

## Questions, Bug Reports and Code Contributions

[Read the contribution guide](https://github.com/cryoem-uoft/cryosparc-tools/blob/develop/CONTRIBUTING.md) for full details.

If you publish an open-source tool that uses this package to GitHub, add the
`cryosparc-tools` topic to your repository so others may discover it.
[Browse tagged packages here](https://github.com/topics/cryosparc-tools).

## License

cryosparc-tools is licensed under the BSD-3-Clause license.
[View full license text](https://github.com/cryoem-uoft/cryosparc-tools/blob/main/LICENSE).

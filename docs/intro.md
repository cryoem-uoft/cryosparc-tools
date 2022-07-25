# Getting Started

`cryosparc-tools` is a Python library that enables scripting access to the
[cryoSPARC](https://cryosparc.com) <abbr title="cryo-electron microscopy">cryo-EM</abbr> software package.

Use it for the following use cases:

* Programmatically read and write exposure, particle and volume data
* Access project and job metadata
* Extend cryoSPARC functionality with third-party software packages

```{note}
This guide documents usage of the `cryosparc-tools` Python library. For cryoSPARC installation or general cryoSPARC usage instructions, [read the official guide](https://guide.cryosparc.com).
```


## Pre-requisites

* [Python >= 3.7](https://www.python.org/downloads/)
* [cryoSPARC >= v4.0](https://cryosparc.com/download)

cryoSPARC installation must be accessible via one of the following methods:
- Running on the local machine
- Running on a machine on the same network with `BASE_PORT + 2` and `BASE_PORT + 3` open for TCP connections
- Running on a remote machine with `BASE_PORT + 2` and `BASE_PORT + 3` forwarded to the local machine

See [SSH Port Forwarding](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/accessing-cryosparc#ssh-port-forwarding-on-a-nix-system)
documentation for accessing a cryoSPARC instance on a remote machine accessible
via <abbr title="Secure Shell">SSH</abbr>.



## Installation

Install `cryosparc-tools` via pip:

```sh
pip install cryosparc-tools
```

Or Conda:

```sh
conda install -c conda-forge cryosparc-tools
```

## Usage

Import in a Python module and connect to a CryoSPARC instance

```py
from cryosparc.tools import CryoSPARC

cs = CryoSPARC(port=39000)
assert cs.test_connection()
```

```{note}
This assumes cryoSPARC base ports +2 and +3 (e.g., 39002 and 39003) are available at `localhost` on the local machine. If cryoSPARC is on another machine on the same network, use `CryoSPARC(host="hostname", port=39000)`
```

Query projects, jobs and result datasets

```py
project = cs.find_project("P3")
job = project.find_job("J42")
micrographs = job.load_output("exposures")

for mic in micrographs.rows():
    print(mic["blob/path"])
```

Load and save datasets directly (assumes project directory is available on
current machine):

```py
from cryosparc.dataset import Dataset

path = project.dir() / "J43" / "particles.cs"
particles = Dataset.load(path)

for particle in particles.rows:
    shift = particle["alignments2D/shift"]
    particle["location/center_x_frac"] += shift[0]
    particle["location/center_y_frac"] += shift[1]

particles.save(path)
```

## Next Steps

Browse the included examples real-life use cases for `cryosparc-tools`. Read the
API Reference for full usage capabilities.

```{tableofcontents}
```

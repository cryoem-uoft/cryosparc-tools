# Getting Started

`cryosparc-tools` is a Python library that enables scripting access to the
[CryoSPARC](https://cryosparc.com) <abbr title="cryo-electron microscopy">cryo-EM</abbr> software package.

Use it for the following use cases:

- Programmatically read and write exposure, particle and volume data
- Access project and job metadata
- Extend CryoSPARC functionality with third-party software packages

```{note}
This guide documents usage of the `cryosparc-tools` Python library. For CryoSPARC installation or general CryoSPARC usage instructions, [read the official guide](https://guide.cryosparc.com).
```

## Pre-requisites

- [Python >= 3.7](https://www.python.org/downloads/)
- [CryoSPARC >= v4.1](https://cryosparc.com/download)

CryoSPARC installation must be accessible via one of the following methods:

- Running on the local machine
- Running on a machine on the same network with `BASE_PORT + 2` and `BASE_PORT + 3` open for TCP connections
- Running on a remote machine with `BASE_PORT + 2` and `BASE_PORT + 3` forwarded to the local machine

See [SSH Port Forwarding](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/accessing-cryosparc#ssh-port-forwarding-on-a-nix-system)
documentation for accessing a CryoSPARC instance on a remote machine accessible
via <abbr title="Secure Shell">SSH</abbr>.

## Installation

Install `cryosparc-tools` from PyPI:

```sh
pip install cryosparc-tools
```

## Usage

Import from a Python module and connect to a CryoSPARC instance. Include your
CryoSPARC license ID, the network hostname of the machine hosting your CryoSPARC
instance, the instance's base port number and your email/password login
credentials.

```py
from cryosparc.tools import CryoSPARC

cs = CryoSPARC(
    license='xxxxxxxx-xxxx-xxxx-xxxxxxxxxxxx',
    hostname='localhost',
    port=39000,
    email='ali@example.com',
    password='password123'
)
```

```{note}
The example above assumes CryoSPARC base ports +2 and +3 (e.g., 39002 and 39003)
are available at `localhost` on the local machine. If CryoSPARC is on another
machine on the same network, use `CryoSPARC(host="hostname", port=39000)`
```

Query projects, jobs and result datasets:

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
    shift_y, shift_x = particle["alignments2D/shift"].T
    mic_shape_y, mic_shape_x = particles["location/micrograph_shape"].T
    new_loc_x = particles["location/center_x_frac"] * mic_shape_x - shift_x
    new_loc_y = particles["location/center_y_frac"] * mic_shape_y - shift_y
    particle["location/center_x_frac"] *= new_loc_x / mic_shape_x
    particle["location/center_y_frac"] *= new_loc_y / mic_shape_y

particles.save(path)
```

## Next Steps

Browse the included examples real-life use cases for `cryosparc-tools`. Read the
API Reference for full usage capabilities.

```{tableofcontents}

```

## Contributing

For questions, bug reports, suggestions or source code contributions, please
[read the contribution guide](https://github.com/cryoem-uoft/cryosparc-tools/blob/main/CONTRIBUTING.md).

If you publish an open-source tool that uses this package to GitHub, add the `cryosparc-tools` topic to your repository so others may discover it. [Browse tagged packages here](https://github.com/topics/cryosparc-tools).

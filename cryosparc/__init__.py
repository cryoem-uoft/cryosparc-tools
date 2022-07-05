__version__ = "0.1.0"

from typing import Optional

from cryosparc.project import Project

from .command import CommandClient
from .dataset import Dataset
from .project import Project

assert Dataset


class CryoSPARC:
    """
    High-level class for interfacing with a running cryoSPARC instance.

    Initialize with the host and base port of the running cryoSPARC instance
    accessible on the current network.

    Example usage:

    ```
    from cryosparc import CryoSPARC

    cs = CryoSPARC(port=39000)
    project = cs.find_project('P3')
    job = project.find_job('J42')
    micrographs = job.load_output('exposures')

    # Remove corrupt exposures
    filtered_micrographs = micrographs.query(is_mic_corrupt)
    job.save_output('micrographs', filtered_micrographs)
    ```

    """

    def __init__(self, host: str = "localhost", port: int = 39000) -> None:
        self.cli = CommandClient(host=host, port=port + 2)

    def find_project(self, project_uid: str) -> Optional[Project]:
        pass

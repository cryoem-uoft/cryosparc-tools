from contextlib import contextmanager
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Iterable, Optional, Union
from typing_extensions import Literal

import numpy.typing as nt

from .spec import Datatype, Datafield, Datatype
from .dataset import Dataset

if TYPE_CHECKING:
    from .tools import CryoSPARC


class Job:
    """
    Immutable reference to a job in cryoSPARC with ability to load inputs and
    outputs
    """

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.puid = project_uid
        self.juid = uid

    def dir(self) -> PurePosixPath:
        """
        Get the path to the job directory
        """
        return NotImplemented

    def load_input(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        return NotImplemented

    def load_output(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        return NotImplemented

    def save_output(self, name: str, dataset: Dataset):
        return NotImplemented

    def log(self, text, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log
        """
        pass

    def download(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.juid) / path
        return self.cs.download(self.puid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        path = PurePosixPath(self.juid) / path
        return self.cs.download_file(self.puid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.juid) / path
        return self.cs.download_dataset(self.puid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.juid) / path
        return self.cs.download_mrc(self.puid, path)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        path = PurePosixPath(self.juid) / path
        return self.cs.upload(self.puid, path, file)

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        path = PurePosixPath(self.juid) / path
        return self.cs.upload_dataset(self.puid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: nt.NDArray, psize: float):
        path = PurePosixPath(self.juid) / path
        return self.cs.upload_mrc(self.puid, path, data, psize)


class CustomJob(Job):
    """
    Mutable custom job with customizeble input slots and saveable results
    """

    def add_input(
        self,
        name: str,
        type: Datatype,
        title: Optional[str] = None,
        fields: Iterable[Union[str, Datafield]] = [],
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
    ):
        """
        Add an input slot to the current job.
        """
        pass

    def add_output(
        self,
        name: str,
        type: Optional[Datatype] = None,
        inherits: Optional[str] = None,
        title: Optional[str] = None,
        fields: Iterable[Union[str, Datafield]] = [],
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
    ):
        """
        Add an output slot to the current job.

        One of `type` or `inherits` must be specified, where `inherits` is the
        name of an existing input.
        """
        pass

    def connect(self, input: str, job_uid: str, output: str):
        """
        Connect the given input to an output with given job UID and name
        """
        pass

    @contextmanager
    def run(self):
        # TODO: Set job to running status
        try:
            yield self
            # TODO: Set job to completed status
        except:
            # TODO: Set job to error status
            pass

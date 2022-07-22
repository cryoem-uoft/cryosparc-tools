from contextlib import contextmanager
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Iterable, List, Optional, Tuple, Union
from typing_extensions import Literal, TypedDict

import numpy.typing as nt

from .spec import Datatype, Datafield, Datatype
from .dataset import Dataset

if TYPE_CHECKING:
    from .tools import CryoSPARC


class OutputResult(TypedDict):
    uid: str
    type: str
    name: str
    group_name: str
    title: str
    description: str
    versions: List[int]
    metafiles: List[str]
    min_fields: List[Tuple[str, str]]
    num_items: int
    passthrough: bool


class JobDoc(TypedDict):
    output_results: List[OutputResult]


class Job:
    """
    Immutable reference to a job in cryoSPARC with ability to load inputs and
    outputs
    """

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.project_uid = project_uid
        self.uid = uid

    @property
    def doc(self) -> JobDoc:
        if not self._doc:
            self.refresh()
        return self._doc

    def refresh(self):
        self._doc = self.cs.cli.get_job(self.project_uid, self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        return PurePosixPath(self.cs.cli.get_job_dir_abs(self.project_uid, self.uid))  # type: ignore

    def load_input(self, name: str, fields: Iterable[str] = []) -> Dataset:
        return NotImplemented

    def load_output(self, name: str, fields: Iterable[str] = []) -> Dataset:
        job = self.doc
        fields = set(fields)
        results = [
            result
            for result in job["output_results"]
            if result["group_name"] == name and (not fields or result["name"] in fields)
        ]
        if not results:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have any results for output {name}")

        metafiles = set().union(*(r["metafiles"] for r in results))
        datasets = [self.cs.download_dataset(self.project_uid, f) for f in metafiles]
        return Dataset.innerjoin(*datasets)

    def save_output(self, name: str, dataset: Dataset):
        return NotImplemented

    def log(self, text, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log
        """
        pass

    def download(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download(self.project_uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_file(self.project_uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_dataset(self.project_uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_mrc(self.project_uid, path)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        path = PurePosixPath(self.uid) / path
        return self.cs.upload(self.project_uid, path, file)

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        path = PurePosixPath(self.uid) / path
        return self.cs.upload_dataset(self.project_uid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: nt.NDArray, psize: float):
        path = PurePosixPath(self.uid) / path
        return self.cs.upload_mrc(self.project_uid, path, data, psize)


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

import re
from contextlib import contextmanager
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Iterable, List, Optional, Pattern, TextIO, Union
from typing_extensions import Literal

import numpy.typing as nt

from cryosparc.dtype import decode_fields

from .spec import Datatype, Datafield, Datatype, JobDocument
from .util import first
from .dataset import Dataset

if TYPE_CHECKING:
    from .tools import CryoSPARC


class JobLogIO(TextIO):
    def __init__(
        self,
        cs: "CryoSPARC",
        project_uid: str,
        job_uid: str,
        checkpoint_line_pattern: Union[str, Pattern[str], Literal[None]] = None,
        pipe: Union[TextIO, None] = None,
    ):
        super().__init__()
        self.cs = cs
        self.project_uid = project_uid
        self.job_uid = job_uid

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def write(self, __s: str) -> int:
        return super().write(__s)


class Job:
    """
    Immutable reference to a job in cryoSPARC with ability to load inputs and
    outputs
    """

    _doc: Optional[JobDocument] = None

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.project_uid = project_uid
        self.uid = uid

    @property
    def doc(self) -> JobDocument:
        if not self._doc:
            self.refresh()
        assert self._doc, "Could not refresh job document"
        return self._doc

    def refresh(self):
        self._doc = self.cs.cli.get_job(self.project_uid, self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        return PurePosixPath(self.cs.cli.get_job_dir_abs(self.project_uid, self.uid))  # type: ignore

    def load_input(self, name: str, fields: Iterable[str] = []):
        job = self.doc
        group = first(s for s in job["input_slot_groups"] if s["name"] == name)
        if not group:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have an input {name}")

        data = {"project_uid": self.project_uid, "job_uid": self.uid, "input_name": name, "slots": list(fields)}

        with self.cs.vis._json_request("/load_job_input", data=data) as response:
            mime = response.headers.get("Content-Type")
            if mime != "application/x-cryosparc-dataset":
                raise TypeError(f"Unable to load dataset for job {self.project_uid}-{self.uid} input {name}")
            return Dataset.load(response)

    def load_output(self, name: str, fields: Iterable[str] = []):
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

    def log(self, text, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log
        """
        pass

    def logio(
        self,
        level: Literal["text", "warning", "error"] = "text",
        checkpoint_line_pattern: Union[str, Pattern[str], Literal[None]] = None,
        pipe: Union[TextIO, None] = None,
    ) -> TextIO:
        """
        Get a writeable handle with the same interface as sys.stdout or
        sys.stderr that when written to writes to this job's streamlong
        """
        if checkpoint_line_pattern and not isinstance(checkpoint_line_pattern, re.Pattern):
            checkpoint_line_pattern = re.compile(checkpoint_line_pattern)

        return NotImplemented

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

    def subprocess(
        self,
        args: list,
        checkpoint_line_pattern: Union[str, Pattern[str], Literal[None]] = None,
        mute_stdout: bool = False,
        mute_stderr: bool = False,
        **kwargs,
    ):
        """
        Launch a subprocess and write its output and error to the job log.
        """
        import subprocess
        import sys

        args = list(map(str, args))
        return subprocess.run(args, **kwargs)


class ExternalJob(Job):
    """
    Mutable custom job with customizeble input slots and saveable results. Used
    to save data
    """

    def add_input(
        self,
        type: Datatype,
        name: Optional[str] = None,
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
        slots: Iterable[Union[str, Datafield]] = [],
        title: Optional[str] = None,
    ):
        """
        Add an input slot to the current job.
        """
        self.cs.vis.add_external_job_input(  # type: ignore
            project_uid=self.project_uid,
            job_uid=self.uid,
            type=type,
            name=name,
            min=min,
            max=max,
            slots=slots,
            title=title,
        )
        self.refresh()
        return self.doc["output_result_groups"][-1]["name"]

    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = None,
        slots: List[Union[str, Datafield]] = [],
        passthrough: Union[str, Literal[False]] = False,
        title: Optional[str] = None,
    ):
        """
        Add an output slot to the current job.

        One of `type` or `passthrough` must be specified, where `passthrough` is
        the name of an existing input (added via `add_input`).

        Returns the name of the created output
        """
        self.cs.vis.add_external_job_output(  # type: ignore
            project_uid=self.project_uid,
            job_uid=self.uid,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough,
            title=title,
        )
        self.refresh()
        return self.doc["output_result_groups"][-1]["name"]

    def connect(
        self,
        source_job_uid: str,
        source_output: str,
        target_input: str,
        slots: List[Union[str, Datafield]] = [],
        title: str = "",
        desc: str = "",
    ):
        """
        Connect the given input for this job to an output with given job UID and
        name. If this input does not exist, it will be added with the given
        slots. At least one slot must be specified if the input does not exist.
        """
        assert source_job_uid != self.uid, f"Cannot connect job {self.uid} to itself"
        status: bool = self.cs.vis.connect_external_job(  # type: ignore
            project_uid=self.project_uid,
            source_job_uid=source_job_uid,
            source_output=source_output,
            target_job_uid=self.uid,
            target_input=target_input,
            slots=slots,
            title=title,
            desc=desc,
        )
        self.refresh()
        return status

    def init_output(self, name: str, size: int = 0):
        """
        Allocate an empty dataset for the given output with the given name.
        Initialize with the given number of empty rows.
        """
        fields = self.cs.cli.job_output_fields(self.project_uid, self.uid, name)  # type: ignore
        fields = decode_fields(fields)
        return Dataset.allocate(size, fields)

    def save_output(self, name: str, dataset: Dataset):
        """
        Job must have status "running" for this to work
        """
        url = f"/external/upload/{self.project_uid}/{self.uid}/{name}"
        with self.cs.vis._request(url, data=dataset.stream()) as res:
            result = res.read().decode()
            assert res.status >= 200 and res.status < 400, f"Save output failed with message: {result}"

    def start(self, status: Literal["running", "waiting"] = "waiting"):
        # Set job status to "running"
        assert status in {"running", "waiting"}, f"Invalid start status {status}"
        self.cs.cli.set_job_status(self.project_uid, self.uid, status)  # type: ignore

    def stop(self, error=False):
        # Set job status to "completed" or "failed"
        status = "failed" if error else "completed"
        self.cs.cli.set_job_status(self.project_uid, self.uid, status)  # type: ignore

    @contextmanager
    def run(self):
        # TODO: Set job to running status
        error = False
        self.start("running")
        try:
            yield self
        # TODO: Set job to completed status
        except:
            # TODO: Set job to error status, send error to joblog
            error = True
            raise
        finally:
            self.stop(error)

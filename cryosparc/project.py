from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Optional, Union

import numpy.typing as nt

from .dataset import Dataset
from .job import Job, CustomJob

if TYPE_CHECKING:
    from .tools import CryoSPARC


class Project:
    def __init__(self, cs: "CryoSPARC", uid: str) -> None:
        self.cs = cs
        self.uid = uid
        self._doc = {}

    @property
    def doc(self) -> dict:
        if not self._doc:
            self.refresh()
        return self._doc

    def refresh(self):
        self._doc = self.cs.cli.get_project(self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        return PurePosixPath(self.cs.cli.get_project_dir_abs(self.uid))  # type: ignore

    def find_job(self, job_uid: str) -> Job:
        job = Job(self.cs, self.uid, job_uid)
        job.refresh()
        return job

    def find_custom_job(self, job_uid: str) -> Optional[CustomJob]:
        job = CustomJob(self.cs, self.uid, job_uid)
        job.refresh()
        return job

    def create_custom_job(self) -> Optional[CustomJob]:
        pass

    def download(self, path: Union[str, PurePosixPath]):
        return self.cs.download(self.uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        return self.cs.download_file(self.uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        return self.cs.download_dataset(self.uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        return self.cs.download_mrc(self.uid, path)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        return self.cs.upload(self.uid, path, file)

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        return self.cs.upload_dataset(self.uid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: nt.NDArray, psize: float):
        return self.cs.upload_mrc(self.uid, path, data, psize)

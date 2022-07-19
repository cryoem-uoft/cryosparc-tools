from pathlib import PurePath, PurePosixPath
from typing import IO, Optional, Union

import numpy.typing as nt

from .dataset import Dataset
from .job import Job, CustomJob
from .util import bopen
from . import CryoSPARC


class Project:
    def __init__(self, cs: CryoSPARC, uid: str) -> None:
        self.cs = cs
        self.puid = uid

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        return NotImplemented

    def find_job(self, job_uid: str) -> Optional[Job]:
        try:
            return Job(self.cs, self.puid, job_uid)
        except:
            return None

    def find_custom_job(self, job_uid: str) -> Optional[CustomJob]:
        try:
            return CustomJob(self.cs, self.puid, job_uid)
        except:
            return None

    def create_custom_job(self) -> Optional[CustomJob]:
        pass

    def download(self, path: Union[str, PurePosixPath]):
        return self.cs.download(self.puid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        return self.cs.download_file(self.puid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        return self.cs.download_dataset(self.puid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        return self.cs.download_mrc(self.puid, path)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        return self.cs.upload(self.puid, path, file)

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        return self.cs.upload_dataset(self.puid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: nt.NDArray, psize: float):
        return self.cs.upload_mrc(self.puid, path, data, psize)
